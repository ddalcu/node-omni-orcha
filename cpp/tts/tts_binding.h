#pragma once
#include <napi.h>
#include <string>
#include <memory>

// Forward declarations — TTS.cpp types
struct tts_generation_runner;

class TtsContext : public Napi::ObjectWrap<TtsContext> {
public:
  TtsContext(const Napi::CallbackInfo& info);
  ~TtsContext();

  Napi::Value Speak(const Napi::CallbackInfo& info);
  Napi::Value ListVoices(const Napi::CallbackInfo& info);
  Napi::Value Unload(const Napi::CallbackInfo& info);

private:
  void Cleanup();

  std::unique_ptr<tts_generation_runner> runner_;
  std::string model_path_;
  float sampling_rate_ = 44100.0f;
};
