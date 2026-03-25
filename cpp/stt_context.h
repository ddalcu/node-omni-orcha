#pragma once
#include <napi.h>
#include "whisper.h"
#include <string>
#include <vector>

class SttContext : public Napi::ObjectWrap<SttContext> {
public:
  SttContext(const Napi::CallbackInfo& info);
  ~SttContext();

  Napi::Value Transcribe(const Napi::CallbackInfo& info);
  Napi::Value DetectLanguage(const Napi::CallbackInfo& info);
  Napi::Value Unload(const Napi::CallbackInfo& info);

  static void Register(Napi::Env env, Napi::Object& exports);
  static Napi::FunctionReference constructor_;

private:
  void Cleanup();

  whisper_context* ctx_ = nullptr;
  std::string model_path_;
};
