#include "tts_context.h"
#include "common/error_helpers.h"
#include <string>
#include <vector>
#include <cstring>
#include <cmath>

Napi::FunctionReference TtsContext::constructor_;

// WAV header writer (16-bit PCM mono)
static std::vector<uint8_t> encode_wav(const float* data, size_t n_samples, int sample_rate) {
  uint16_t channels = 1;
  uint16_t bits_per_sample = 16;
  uint32_t byte_rate = sample_rate * channels * bits_per_sample / 8;
  uint16_t block_align = channels * bits_per_sample / 8;
  uint32_t data_size = (uint32_t)(n_samples * block_align);
  uint32_t file_size = 36 + data_size;

  std::vector<uint8_t> wav;
  wav.reserve(44 + data_size);

  auto write32 = [&](uint32_t v) {
    wav.push_back(v & 0xFF); wav.push_back((v >> 8) & 0xFF);
    wav.push_back((v >> 16) & 0xFF); wav.push_back((v >> 24) & 0xFF);
  };
  auto write16 = [&](uint16_t v) {
    wav.push_back(v & 0xFF); wav.push_back((v >> 8) & 0xFF);
  };
  auto writeStr = [&](const char* s) {
    for (int i = 0; i < 4; i++) wav.push_back(s[i]);
  };

  writeStr("RIFF"); write32(file_size); writeStr("WAVE");
  writeStr("fmt "); write32(16); write16(1); write16(channels);
  write32(sample_rate); write32(byte_rate); write16(block_align); write16(bits_per_sample);
  writeStr("data"); write32(data_size);

  for (size_t i = 0; i < n_samples; i++) {
    float sample = std::fmax(-1.0f, std::fmin(1.0f, data[i]));
    int16_t s16 = (int16_t)(sample * 32767.0f);
    wav.push_back(s16 & 0xFF);
    wav.push_back((s16 >> 8) & 0xFF);
  }
  return wav;
}

// ─── Constructor ───

TtsContext::TtsContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<TtsContext>(info) {
  auto env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Model directory path (string) required");
  }

  std::string modelDir = info[0].As<Napi::String>().Utf8Value();
  if (modelDir.empty()) {
    throw Napi::Error::New(env, "Model directory path must not be empty");
  }
  int threads = 4;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    threads = getInt32Option(opts, "threads", 4);
  }

  tts_ = qwen3_tts_create(modelDir.c_str(), threads);
  if (!tts_) {
    throw Napi::Error::New(env, std::string("Failed to create Qwen3 TTS from: ") + modelDir);
  }
}

TtsContext::~TtsContext() {
  if (tts_) {
    qwen3_tts_destroy(tts_);
    tts_ = nullptr;
  }
}

// ─── Speak ───

class SpeakWorker : public Napi::AsyncWorker {
public:
  SpeakWorker(
    Napi::Env env,
    Qwen3Tts* tts,
    std::string text,
    std::string referenceAudioPath,
    Qwen3TtsParams params
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      tts_(tts),
      text_(std::move(text)),
      referenceAudioPath_(std::move(referenceAudioPath)),
      params_(params) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    Qwen3TtsAudio* audio = nullptr;
    if (!referenceAudioPath_.empty()) {
      audio = qwen3_tts_synthesize_with_voice_file(
        tts_, text_.c_str(), referenceAudioPath_.c_str(), &params_);
    } else {
      audio = qwen3_tts_synthesize(tts_, text_.c_str(), &params_);
    }

    if (!audio || !audio->samples || audio->n_samples == 0) {
      const char* err = qwen3_tts_get_error(tts_);
      std::string msg = "Qwen3 TTS generation failed";
      if (err && strlen(err) > 0) msg += std::string(": ") + err;
      if (audio) qwen3_tts_free_audio(audio);
      SetError(msg);
      return;
    }

    wav_data_ = encode_wav(audio->samples, audio->n_samples, audio->sample_rate);
    qwen3_tts_free_audio(audio);
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);
    auto buffer = Napi::Buffer<uint8_t>::Copy(env, wav_data_.data(), wav_data_.size());
    deferred_.Resolve(buffer);
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  Qwen3Tts* tts_;
  std::string text_;
  std::string referenceAudioPath_;
  Qwen3TtsParams params_;
  std::vector<uint8_t> wav_data_;
};

Napi::Value TtsContext::Speak(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (!tts_) throw Napi::Error::New(env, "TTS not loaded");
  if (info.Length() < 1 || !info[0].IsString())
    throw Napi::TypeError::New(env, "Text string required");

  std::string text = info[0].As<Napi::String>().Utf8Value();
  if (text.empty()) {
    throw Napi::Error::New(env, "Text must not be empty");
  }
  std::string referenceAudioPath = "";

  Qwen3TtsParams params;
  qwen3_tts_default_params(&params);

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    params.temperature = (float)getDoubleOption(opts, "temperature", 0.9);
    params.top_k = getInt32Option(opts, "topK", 50);
    params.max_audio_tokens = getInt32Option(opts, "maxTokens", 4096);
    params.repetition_penalty = (float)getDoubleOption(opts, "repetitionPenalty", 1.05);

    if (opts.Has("referenceAudioPath") && opts.Get("referenceAudioPath").IsString()) {
      referenceAudioPath = opts.Get("referenceAudioPath").As<Napi::String>().Utf8Value();
    }
  }

  auto* worker = new SpeakWorker(env, tts_, std::move(text), std::move(referenceAudioPath), params);
  worker->Queue();
  return worker->Promise();
}

// ─── Unload ───

Napi::Value TtsContext::Unload(const Napi::CallbackInfo& info) {
  if (tts_) {
    qwen3_tts_destroy(tts_);
    tts_ = nullptr;
  }
  auto deferred = Napi::Promise::Deferred::New(info.Env());
  deferred.Resolve(info.Env().Undefined());
  return deferred.Promise();
}

// ─── Registration ───

static Napi::Value CreateTtsContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  try {
    std::vector<napi_value> args;
    for (size_t i = 0; i < info.Length(); i++) args.push_back(info[i]);
    Napi::Object ctx = TtsContext::constructor_.New(args);
    deferred.Resolve(ctx);
  } catch (const Napi::Error& e) {
    deferred.Reject(e.Value());
  }
  return deferred.Promise();
}

void TtsContext::Register(Napi::Env env, Napi::Object& exports) {
  Napi::Function func = DefineClass(env, "TtsContext", {
    InstanceMethod("speak", &TtsContext::Speak),
    InstanceMethod("unload", &TtsContext::Unload),
  });

  constructor_ = Napi::Persistent(func);
  constructor_.SuppressDestruct();

  exports.Set("createTtsContext", Napi::Function::New(env, CreateTtsContext));
}
