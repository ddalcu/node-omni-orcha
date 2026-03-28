#pragma once
#include <napi.h>
#include "llama.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <string>
#include <vector>
#include <atomic>
#include <memory>

// Image data extracted from multimodal messages on the JS thread
struct ImageInput {
  std::vector<unsigned char> data; // raw file bytes (PNG/JPEG/etc)
  uint32_t width = 0;   // only set for raw RGB
  uint32_t height = 0;  // only set for raw RGB
  bool is_raw_rgb = false;
};

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
  Napi::Value HasVision(const Napi::CallbackInfo& info);

  std::vector<common_chat_msg> ParseMessages(const Napi::Array& arr);
  std::vector<common_chat_tool> ParseTools(const Napi::Array& arr);

  // Parse multimodal messages — returns flattened text with image markers,
  // and populates images vector with extracted image data.
  std::string ParseMultimodalMessages(
    const Napi::Array& arr,
    std::vector<common_chat_msg>& messages,
    std::vector<ImageInput>& images);

  // Acquire exclusive access for an inference operation. Returns false if busy.
  bool AcquireInference();
  void ReleaseInference();

  // Register this class with a Napi module exports object
  static void Register(Napi::Env env, Napi::Object& exports);

  // Shared constructor reference for factory function
  static Napi::FunctionReference constructor_;

  // Vision capability
  bool has_vision() const { return mtmd_ctx_ != nullptr; }
  mtmd_context* mtmd_ctx() const { return mtmd_ctx_; }

private:
  void Cleanup();

  llama_model* model_ = nullptr;
  llama_context* ctx_ = nullptr;
  mtmd_context* mtmd_ctx_ = nullptr;
  common_chat_templates_ptr chat_templates_;
  std::atomic<bool> abort_flag_{false};
  std::atomic<bool> busy_{false};
  int n_ctx_ = 0;
};
