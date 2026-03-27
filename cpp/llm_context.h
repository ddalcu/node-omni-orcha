#pragma once
#include <napi.h>
#include "llama.h"
#include "chat.h"
#include <string>
#include <vector>
#include <atomic>
#include <memory>

class LlmContext : public Napi::ObjectWrap<LlmContext> {
public:
  LlmContext(const Napi::CallbackInfo& info);
  ~LlmContext();

  Napi::Value Complete(const Napi::CallbackInfo& info);
  Napi::Value Stream(const Napi::CallbackInfo& info);
  Napi::Value Embed(const Napi::CallbackInfo& info);
  Napi::Value EmbedBatch(const Napi::CallbackInfo& info);
  Napi::Value Unload(const Napi::CallbackInfo& info);
  Napi::Value Abort(const Napi::CallbackInfo& info);

  std::vector<common_chat_msg> ParseMessages(const Napi::Array& arr);
  std::vector<common_chat_tool> ParseTools(const Napi::Array& arr);

  // Acquire exclusive access for an inference operation. Returns false if busy.
  bool AcquireInference();
  void ReleaseInference();

  // Register this class with a Napi module exports object
  static void Register(Napi::Env env, Napi::Object& exports);

  // Shared constructor reference for factory function
  static Napi::FunctionReference constructor_;

private:
  void Cleanup();

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  common_chat_templates_ptr chat_templates_;
  std::atomic<bool> abort_flag_{false};
  std::atomic<bool> busy_{false};
  int n_ctx_ = 0;
};
