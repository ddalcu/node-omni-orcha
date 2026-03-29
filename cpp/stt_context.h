#pragma once
#include <napi.h>
#include "whisper.h"
#include <string>
#include <vector>
#include <atomic>

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

  bool AcquireBusy() {
    bool expected = false;
    return busy_.compare_exchange_strong(expected, true);
  }
  void ReleaseBusy() { busy_.store(false); }

  whisper_context* ctx_ = nullptr;
  std::string model_path_;
  std::atomic<bool> busy_{false};
};
