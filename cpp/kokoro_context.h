#pragma once
#include <napi.h>
#include "kokoro.h"
#include <string>
#include <vector>

class KokoroContext : public Napi::ObjectWrap<KokoroContext> {
public:
  KokoroContext(const Napi::CallbackInfo& info);
  ~KokoroContext();

  Napi::Value Speak(const Napi::CallbackInfo& info);
  Napi::Value ListVoices(const Napi::CallbackInfo& info);
  Napi::Value Unload(const Napi::CallbackInfo& info);

  static void Register(Napi::Env env, Napi::Object& exports);
  static Napi::FunctionReference constructor_;

private:
  kokoro_context* ctx_ = nullptr;
};
