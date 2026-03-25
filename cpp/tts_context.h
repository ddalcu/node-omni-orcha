#pragma once
#include <napi.h>
#include "qwen3tts_c_api.h"
#include <string>
#include <vector>

class TtsContext : public Napi::ObjectWrap<TtsContext> {
public:
  TtsContext(const Napi::CallbackInfo& info);
  ~TtsContext();

  Napi::Value Speak(const Napi::CallbackInfo& info);
  Napi::Value Unload(const Napi::CallbackInfo& info);

  static void Register(Napi::Env env, Napi::Object& exports);
  static Napi::FunctionReference constructor_;

private:
  Qwen3Tts* tts_ = nullptr;
};
