#pragma once
#include <napi.h>
#include "stable-diffusion.h"
#include <string>
#include <atomic>

class InitWorker;

class ImageContext : public Napi::ObjectWrap<ImageContext> {
public:
  ImageContext(const Napi::CallbackInfo& info);
  ~ImageContext();

  Napi::Value Generate(const Napi::CallbackInfo& info);
  Napi::Value GenerateVideo(const Napi::CallbackInfo& info);
  Napi::Value Unload(const Napi::CallbackInfo& info);

  static void Register(Napi::Env env, Napi::Object& exports);
  static Napi::FunctionReference constructor_;

private:
  void Cleanup();

  friend class InitWorker;

  sd_ctx_t* ctx_ = nullptr;
  std::string model_path_;
  std::string clip_l_path_;
  std::string t5xxl_path_;
  std::string llm_path_;
  std::string vae_path_;
  std::string high_noise_diffusion_model_path_;
  int threads_ = -1;
  bool keep_vae_on_cpu_ = false;
  bool offload_to_cpu_ = false;
  bool flash_attn_ = true;
  bool vae_decode_only_ = true;
};
