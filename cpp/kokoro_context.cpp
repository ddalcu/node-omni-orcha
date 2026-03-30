#include "kokoro_context.h"
#include "common/error_helpers.h"
#include <string>
#include <vector>
#include <cstring>
#include <cmath>

Napi::FunctionReference KokoroContext::constructor_;

// WAV header writer (16-bit PCM mono) — same as tts_context.cpp
static std::vector<uint8_t> encode_wav_kokoro(const float* data, size_t n_samples, int sample_rate) {
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

  // Normalize — Kokoro outputs beyond ±1.0 for many voices, and occasionally
  // produces extreme outlier samples (1e30+) that would crush the signal if
  // used for peak. Strategy: clamp outliers first, then peak-normalize.
  std::vector<float> buf(n_samples);
  for (size_t i = 0; i < n_samples; i++) {
    float v = data[i];
    if (v > 10.0f) v = 10.0f;
    if (v < -10.0f) v = -10.0f;
    buf[i] = v;
  }

  float peak = 0.0f;
  for (size_t i = 0; i < n_samples; i++) {
    float a = buf[i] > 0 ? buf[i] : -buf[i];
    if (a > peak) peak = a;
  }
  float scale = (peak > 0.001f) ? 0.9f / peak : 1.0f;

  for (size_t i = 0; i < n_samples; i++) {
    float sample = buf[i] * scale;
    if (sample > 1.0f) sample = 1.0f;
    if (sample < -1.0f) sample = -1.0f;
    int16_t s16 = (int16_t)(sample * 32767.0f);
    wav.push_back(s16 & 0xFF);
    wav.push_back((s16 >> 8) & 0xFF);
  }
  return wav;
}

// ─── Constructor ───

KokoroContext::KokoroContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<KokoroContext>(info) {
  auto env = info.Env();

  if (info.Length() < 3 || !info[0].IsString() || !info[1].IsString() || !info[2].IsString()) {
    throw Napi::TypeError::New(env, "Model path, voices path, and dict path (strings) required");
  }

  std::string modelPath = info[0].As<Napi::String>().Utf8Value();
  std::string voicesPath = info[1].As<Napi::String>().Utf8Value();
  std::string dictPath = info[2].As<Napi::String>().Utf8Value();

  if (modelPath.empty() || voicesPath.empty() || dictPath.empty()) {
    throw Napi::Error::New(env, "Model path, voices path, and dict path must not be empty");
  }

  bool use_gpu = true;
  if (info.Length() >= 4 && info[3].IsObject()) {
    Napi::Object opts = info[3].As<Napi::Object>();
    use_gpu = getBoolOption(opts, "useGpu", true);
  }

  ctx_ = kokoro_create(modelPath.c_str(), voicesPath.c_str(), dictPath.c_str(), use_gpu);
  if (!ctx_) {
    throw Napi::Error::New(env, std::string("Failed to create Kokoro TTS context"));
  }
}

KokoroContext::~KokoroContext() {
  if (ctx_) {
    kokoro_free(ctx_);
    ctx_ = nullptr;
  }
}

// ─── Speak ───

class KokoroSpeakWorker : public Napi::AsyncWorker {
public:
  KokoroSpeakWorker(
    Napi::Env env,
    kokoro_context* ctx,
    std::string text,
    std::string voice,
    float speed
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      ctx_(ctx),
      text_(std::move(text)),
      voice_(std::move(voice)),
      speed_(speed) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    kokoro_audio* audio = kokoro_speak(
      ctx_, text_.c_str(),
      voice_.empty() ? nullptr : voice_.c_str(),
      speed_
    );

    if (!audio || !audio->samples || audio->n_samples == 0) {
      const char* err = kokoro_get_error(ctx_);
      std::string msg = "Kokoro TTS generation failed";
      if (err && strlen(err) > 0) msg += std::string(": ") + err;
      if (audio) kokoro_free_audio(audio);
      SetError(msg);
      return;
    }

    wav_data_ = encode_wav_kokoro(audio->samples, audio->n_samples, audio->sample_rate);
    kokoro_free_audio(audio);
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
  kokoro_context* ctx_;
  std::string text_;
  std::string voice_;
  float speed_;
  std::vector<uint8_t> wav_data_;
};

Napi::Value KokoroContext::Speak(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (!ctx_) throw Napi::Error::New(env, "Kokoro not loaded");
  if (info.Length() < 1 || !info[0].IsString())
    throw Napi::TypeError::New(env, "Text string required");

  std::string text = info[0].As<Napi::String>().Utf8Value();
  if (text.empty()) {
    throw Napi::Error::New(env, "Text must not be empty");
  }

  std::string voice = "";
  float speed = 1.0f;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    voice = getStringOption(opts, "voice", "");
    speed = (float)getDoubleOption(opts, "speed", 1.0);
  }

  auto* worker = new KokoroSpeakWorker(env, ctx_, std::move(text),
                                        std::move(voice), speed);
  worker->Queue();
  return worker->Promise();
}

// ─── ListVoices ───

Napi::Value KokoroContext::ListVoices(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (!ctx_) throw Napi::Error::New(env, "Kokoro not loaded");

  int count = 0;
  const char** names = kokoro_list_voices(ctx_, &count);

  auto arr = Napi::Array::New(env, count);
  for (int i = 0; i < count; i++) {
    arr.Set(i, Napi::String::New(env, names[i]));
  }
  return arr;
}

// ─── Unload ───

Napi::Value KokoroContext::Unload(const Napi::CallbackInfo& info) {
  if (ctx_) {
    kokoro_free(ctx_);
    ctx_ = nullptr;
  }
  auto deferred = Napi::Promise::Deferred::New(info.Env());
  deferred.Resolve(info.Env().Undefined());
  return deferred.Promise();
}

// ─── Registration ───

static Napi::Value CreateKokoroContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  auto deferred = Napi::Promise::Deferred::New(env);
  try {
    std::vector<napi_value> args;
    for (size_t i = 0; i < info.Length(); i++) args.push_back(info[i]);
    Napi::Object ctx = KokoroContext::constructor_.New(args);
    deferred.Resolve(ctx);
  } catch (const Napi::Error& e) {
    deferred.Reject(e.Value());
  }
  return deferred.Promise();
}

void KokoroContext::Register(Napi::Env env, Napi::Object& exports) {
  Napi::Function func = DefineClass(env, "KokoroContext", {
    InstanceMethod("speak", &KokoroContext::Speak),
    InstanceMethod("listVoices", &KokoroContext::ListVoices),
    InstanceMethod("unload", &KokoroContext::Unload),
  });

  constructor_ = Napi::Persistent(func);
  constructor_.SuppressDestruct();

  exports.Set("createKokoroContext", Napi::Function::New(env, CreateKokoroContext));
}
