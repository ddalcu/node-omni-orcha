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

private:
  void Cleanup();

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  common_chat_templates_ptr chat_templates_;
  std::atomic<bool> abort_flag_{false};
  int n_ctx_ = 0;
};
