#include "tts_binding.h"
#include "../common/error_helpers.h"
#include "common.h"
#include "loaders.h"
#include <string>
#include <vector>
#include <cstring>
#include <cmath>

// WAV header writer (minimal — 16-bit PCM mono)
static std::vector<uint8_t> encode_wav(const float* data, size_t n_samples, float sample_rate) {
  uint32_t sr = (uint32_t)sample_rate;
  uint16_t channels = 1;
  uint16_t bits_per_sample = 16;
  uint32_t byte_rate = sr * channels * bits_per_sample / 8;
  uint16_t block_align = channels * bits_per_sample / 8;
  uint32_t data_size = (uint32_t)(n_samples * block_align);
  uint32_t file_size = 36 + data_size;

  std::vector<uint8_t> wav;
  wav.reserve(44 + data_size);

  // RIFF header
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

  writeStr("RIFF");
  write32(file_size);
  writeStr("WAVE");

  // fmt chunk
  writeStr("fmt ");
  write32(16);          // chunk size
  write16(1);           // PCM format
  write16(channels);
  write32(sr);
  write32(byte_rate);
  write16(block_align);
  write16(bits_per_sample);

  // data chunk
  writeStr("data");
  write32(data_size);

  // Convert float [-1,1] to int16
  for (size_t i = 0; i < n_samples; i++) {
    float sample = data[i];
    if (sample > 1.0f) sample = 1.0f;
    if (sample < -1.0f) sample = -1.0f;
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
    throw Napi::TypeError::New(env, "Model path (string) required");
  }

  model_path_ = info[0].As<Napi::String>().Utf8Value();

  std::string voice = "";
  int threads = 4;
  bool cpuOnly = true; // TTS.cpp currently has limited GPU support

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    voice = getStringOption(opts, "voice", "");
    threads = getInt32Option(opts, "threads", 4);
    cpuOnly = getBoolOption(opts, "cpuOnly", true);
  }

  generation_configuration config(voice);

  try {
    runner_ = runner_from_file(model_path_.c_str(), threads, config, cpuOnly);
  } catch (const std::exception& e) {
    throw Napi::Error::New(env, std::string("Failed to load TTS model: ") + e.what());
  }

  if (!runner_) {
    throw Napi::Error::New(env, "Failed to load TTS model: " + model_path_);
  }

  sampling_rate_ = runner_->sampling_rate;
}

TtsContext::~TtsContext() {
  Cleanup();
}

void TtsContext::Cleanup() {
  // Note: TTS.cpp's runner destructor has cleanup issues (ggml_free crash).
  // We intentionally leak the runner — the OS reclaims memory on process exit.
  // This is acceptable since models are typically loaded once for the process lifetime.
  if (runner_) {
    runner_.release(); // release ownership without calling destructor
  }
}

// ─── Speak ───

struct SpeakResult {
  std::vector<uint8_t> wav_data;
};

class SpeakWorker : public Napi::AsyncWorker {
public:
  SpeakWorker(
    Napi::Env env,
    tts_generation_runner* runner,
    float sampling_rate,
    std::string text,
    std::string voice,
    float speed,
    float temperature
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      runner_(runner),
      sampling_rate_(sampling_rate),
      text_(std::move(text)),
      voice_(std::move(voice)),
      speed_(speed),
      temperature_(temperature) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    generation_configuration config(voice_);
    config.temperature = temperature_;

    tts_response response;
    response.data = nullptr;
    response.n_outputs = 0;

    try {
      runner_->generate(text_.c_str(), response, config);
    } catch (const std::exception& e) {
      SetError(std::string("TTS generation failed: ") + e.what());
      return;
    }

    if (!response.data || response.n_outputs == 0) {
      SetError("TTS generation produced no audio");
      return;
    }

    // Encode to WAV
    result_.wav_data = encode_wav(response.data, response.n_outputs, sampling_rate_);

    // Free the response data
    free(response.data);
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);

    auto buffer = Napi::Buffer<uint8_t>::Copy(
      env, result_.wav_data.data(), result_.wav_data.size()
    );

    deferred_.Resolve(buffer);
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  tts_generation_runner* runner_;
  float sampling_rate_;
  std::string text_;
  std::string voice_;
  float speed_;
  float temperature_;
  SpeakResult result_;
};

Napi::Value TtsContext::Speak(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!runner_) {
    throw Napi::Error::New(env, "TTS model not loaded");
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Text string required");
  }

  std::string text = info[0].As<Napi::String>().Utf8Value();

  std::string voice = "";
  float temperature = 1.0f;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    voice = getStringOption(opts, "voice", "");
    temperature = (float)getDoubleOption(opts, "temperature", 1.0);
  }

  // Run synchronously — TTS.cpp's ggml graph uses Metal which isn't thread-safe
  generation_configuration config(voice);
  config.temperature = temperature;

  tts_response response;
  response.data = nullptr;
  response.n_outputs = 0;

  runner_->generate(text.c_str(), response, config);

  if (!response.data || response.n_outputs == 0) {
    throw Napi::Error::New(env, "TTS generation produced no audio");
  }

  // Copy to WAV — response.data is owned by the runner, do NOT free it
  auto wav_data = encode_wav(response.data, response.n_outputs, sampling_rate_);

  auto buffer = Napi::Buffer<uint8_t>::Copy(env, wav_data.data(), wav_data.size());

  auto deferred = Napi::Promise::Deferred::New(env);
  deferred.Resolve(buffer);
  return deferred.Promise();
}

// ─── ListVoices ───

Napi::Value TtsContext::ListVoices(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  if (!runner_) {
    throw Napi::Error::New(env, "TTS model not loaded");
  }

  auto voices = runner_->list_voices();
  Napi::Array result = Napi::Array::New(env, voices.size());
  for (size_t i = 0; i < voices.size(); i++) {
    result.Set((uint32_t)i, Napi::String::New(env, std::string(voices[i])));
  }
  return result;
}

// ─── Unload ───

Napi::Value TtsContext::Unload(const Napi::CallbackInfo& info) {
  Cleanup();
  auto deferred = Napi::Promise::Deferred::New(info.Env());
  deferred.Resolve(info.Env().Undefined());
  return deferred.Promise();
}

// ─── Module Factory ───

Napi::Value CreateTtsContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  Napi::Function constructor = env.GetInstanceData<Napi::FunctionReference>()->Value();

  std::vector<napi_value> args;
  for (size_t i = 0; i < info.Length(); i++) {
    args.push_back(info[i]);
  }

  Napi::Object ctx = constructor.New(args);
  auto deferred = Napi::Promise::Deferred::New(env);
  deferred.Resolve(ctx);
  return deferred.Promise();
}

// ─── Module Init ───

Napi::Object InitTtsModule(Napi::Env env, Napi::Object exports) {
  Napi::Function func = TtsContext::DefineClass(env, "TtsContext", {
    TtsContext::InstanceMethod("speak", &TtsContext::Speak),
    TtsContext::InstanceMethod("listVoices", &TtsContext::ListVoices),
    TtsContext::InstanceMethod("unload", &TtsContext::Unload),
  });

  auto* constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);
  constructor->SuppressDestruct();
  env.SetInstanceData(constructor);

  exports.Set("createContext", Napi::Function::New(env, CreateTtsContext));
  return exports;
}

NODE_API_MODULE(tts, InitTtsModule)
