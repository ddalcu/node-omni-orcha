#include "stt_context.h"
#include "common/error_helpers.h"
#include "whisper.h"
#include <string>
#include <vector>
#include <cstring>
#include <cmath>

Napi::FunctionReference SttContext::constructor_;

// ─── Constructor ───

SttContext::SttContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<SttContext>(info) {
  auto env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Model path (string) required");
  }

  model_path_ = info[0].As<Napi::String>().Utf8Value();
  if (model_path_.empty()) {
    throw Napi::Error::New(env, "Model path must not be empty");
  }

  whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = true;
  cparams.flash_attn = true;

  ctx_ = whisper_init_from_file_with_params(model_path_.c_str(), cparams);
  if (!ctx_) {
    throw Napi::Error::New(env, "Failed to load whisper model: " + model_path_);
  }
}

SttContext::~SttContext() {
  Cleanup();
}

void SttContext::Cleanup() {
  if (ctx_) {
    whisper_free(ctx_);
    ctx_ = nullptr;
  }
}

// ─── PCM conversion helper ───

static std::vector<float> pcm16_to_f32(const uint8_t* data, size_t byte_len) {
  size_t n_samples = byte_len / 2;
  std::vector<float> pcmf32(n_samples);
  const int16_t* samples = reinterpret_cast<const int16_t*>(data);
  for (size_t i = 0; i < n_samples; i++) {
    pcmf32[i] = static_cast<float>(samples[i]) / 32768.0f;
  }
  return pcmf32;
}

// ─── Transcribe ───

struct TranscribeResultData {
  std::string text;
  std::string language;
  struct Segment {
    int64_t start_ms;
    int64_t end_ms;
    std::string text;
  };
  std::vector<Segment> segments;
};

class TranscribeWorker : public Napi::AsyncWorker {
public:
  TranscribeWorker(
    Napi::Env env,
    whisper_context* ctx,
    std::vector<float> pcmf32,
    std::string language
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      ctx_(ctx),
      pcmf32_(std::move(pcmf32)),
      language_(std::move(language)) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    bool auto_detect = (language_.empty() || language_ == "auto");
    wparams.language = auto_detect ? nullptr : language_.c_str();
    wparams.detect_language = auto_detect;

    wparams.n_threads = 4;
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.no_timestamps = false;

    int ret = whisper_full(ctx_, wparams, pcmf32_.data(), pcmf32_.size());
    if (ret != 0) {
      SetError("Whisper transcription failed");
      return;
    }

    int n_segments = whisper_full_n_segments(ctx_);

    int lang_id = whisper_full_lang_id(ctx_);
    if (lang_id >= 0) {
      const char* lang_str = whisper_lang_str(lang_id);
      if (lang_str) result_.language = lang_str;
    }

    std::string full_text;
    for (int i = 0; i < n_segments; i++) {
      const char* seg_text = whisper_full_get_segment_text(ctx_, i);
      int64_t t0 = whisper_full_get_segment_t0(ctx_, i);
      int64_t t1 = whisper_full_get_segment_t1(ctx_, i);

      if (seg_text) {
        full_text += seg_text;
        result_.segments.push_back({
          t0 * 10,
          t1 * 10,
          std::string(seg_text),
        });
      }
    }

    result_.text = full_text;
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);

    Napi::Object result = Napi::Object::New(env);
    result.Set("text", Napi::String::New(env, result_.text));
    result.Set("language", Napi::String::New(env, result_.language));

    Napi::Array segments = Napi::Array::New(env, result_.segments.size());
    for (size_t i = 0; i < result_.segments.size(); i++) {
      Napi::Object seg = Napi::Object::New(env);
      seg.Set("start", Napi::Number::New(env, (double)result_.segments[i].start_ms / 1000.0));
      seg.Set("end", Napi::Number::New(env, (double)result_.segments[i].end_ms / 1000.0));
      seg.Set("text", Napi::String::New(env, result_.segments[i].text));
      segments.Set((uint32_t)i, seg);
    }
    result.Set("segments", segments);

    deferred_.Resolve(result);
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  whisper_context* ctx_;
  std::vector<float> pcmf32_;
  std::string language_;
  TranscribeResultData result_;
};

Napi::Value SttContext::Transcribe(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!ctx_) {
    throw Napi::Error::New(env, "Whisper context not loaded");
  }
  if (info.Length() < 1 || !info[0].IsBuffer()) {
    throw Napi::TypeError::New(env, "Audio buffer (16-bit PCM, 16kHz mono) required");
  }

  auto buf = info[0].As<Napi::Buffer<uint8_t>>();
  if (buf.Length() < 2) {
    throw Napi::Error::New(env, "Audio buffer too small — need at least one 16-bit sample (2 bytes)");
  }
  auto pcmf32 = pcm16_to_f32(buf.Data(), buf.Length());

  std::string language = "auto";
  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    language = getStringOption(opts, "language", "auto");
  }

  auto* worker = new TranscribeWorker(env, ctx_, std::move(pcmf32), std::move(language));
  worker->Queue();
  return worker->Promise();
}

// ─── DetectLanguage ───

class DetectLanguageWorker : public Napi::AsyncWorker {
public:
  DetectLanguageWorker(
    Napi::Env env,
    whisper_context* ctx,
    std::vector<float> pcmf32
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      ctx_(ctx),
      pcmf32_(std::move(pcmf32)) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = nullptr;
    wparams.detect_language = true;
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.n_threads = 4;

    size_t max_samples = 16000 * 30;
    size_t n_samples = std::min(pcmf32_.size(), max_samples);

    int ret = whisper_full(ctx_, wparams, pcmf32_.data(), n_samples);
    if (ret != 0) {
      SetError("Language detection failed");
      return;
    }

    int lang_id = whisper_full_lang_id(ctx_);
    if (lang_id >= 0) {
      const char* lang_str = whisper_lang_str(lang_id);
      if (lang_str) {
        detected_lang_ = lang_str;
        return;
      }
    }
    detected_lang_ = "unknown";
  }

  void OnOK() override {
    deferred_.Resolve(Napi::String::New(Env(), detected_lang_));
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  whisper_context* ctx_;
  std::vector<float> pcmf32_;
  std::string detected_lang_;
};

Napi::Value SttContext::DetectLanguage(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!ctx_) {
    throw Napi::Error::New(env, "Whisper context not loaded");
  }
  if (info.Length() < 1 || !info[0].IsBuffer()) {
    throw Napi::TypeError::New(env, "Audio buffer required");
  }

  auto buf = info[0].As<Napi::Buffer<uint8_t>>();
  if (buf.Length() < 2) {
    throw Napi::Error::New(env, "Audio buffer too small — need at least one 16-bit sample (2 bytes)");
  }
  auto pcmf32 = pcm16_to_f32(buf.Data(), buf.Length());

  auto* worker = new DetectLanguageWorker(env, ctx_, std::move(pcmf32));
  worker->Queue();
  return worker->Promise();
}

// ─── Unload ───

Napi::Value SttContext::Unload(const Napi::CallbackInfo& info) {
  Cleanup();
  auto deferred = Napi::Promise::Deferred::New(info.Env());
  deferred.Resolve(info.Env().Undefined());
  return deferred.Promise();
}

// ─── Registration ───

static Napi::Value CreateSttContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  std::vector<napi_value> args;
  for (size_t i = 0; i < info.Length(); i++) {
    args.push_back(info[i]);
  }
  Napi::Object ctx = SttContext::constructor_.New(args);
  auto deferred = Napi::Promise::Deferred::New(env);
  deferred.Resolve(ctx);
  return deferred.Promise();
}

void SttContext::Register(Napi::Env env, Napi::Object& exports) {
  Napi::Function func = DefineClass(env, "SttContext", {
    InstanceMethod("transcribe", &SttContext::Transcribe),
    InstanceMethod("detectLanguage", &SttContext::DetectLanguage),
    InstanceMethod("unload", &SttContext::Unload),
  });

  constructor_ = Napi::Persistent(func);
  constructor_.SuppressDestruct();

  exports.Set("createSttContext", Napi::Function::New(env, CreateSttContext));
}
