#include "image_binding.h"
#include "../common/error_helpers.h"
#include "stable-diffusion.h"

// stb_image_write for PNG encoding
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>
#include <vector>
#include <cstring>

// ─── PNG encoding helper ───

struct PngBuffer {
  std::vector<uint8_t> data;
};

static void png_write_callback(void* context, void* data, int size) {
  auto* buf = static_cast<PngBuffer*>(context);
  auto* bytes = static_cast<uint8_t*>(data);
  buf->data.insert(buf->data.end(), bytes, bytes + size);
}

// ─── Sample method mapping ───

static sample_method_t parse_sample_method(const std::string& name) {
  if (name == "euler") return EULER_SAMPLE_METHOD;
  if (name == "euler_a") return EULER_A_SAMPLE_METHOD;
  if (name == "heun") return HEUN_SAMPLE_METHOD;
  if (name == "dpm2") return DPM2_SAMPLE_METHOD;
  if (name == "dpmpp2s_a") return DPMPP2S_A_SAMPLE_METHOD;
  if (name == "dpmpp2m") return DPMPP2M_SAMPLE_METHOD;
  if (name == "dpmpp2mv2") return DPMPP2Mv2_SAMPLE_METHOD;
  if (name == "lcm") return LCM_SAMPLE_METHOD;
  return EULER_SAMPLE_METHOD;
}

static scheduler_t parse_scheduler(const std::string& name) {
  if (name == "discrete") return DISCRETE_SCHEDULER;
  if (name == "karras") return KARRAS_SCHEDULER;
  if (name == "exponential") return EXPONENTIAL_SCHEDULER;
  if (name == "ays") return AYS_SCHEDULER;
  if (name == "sgm_uniform") return SGM_UNIFORM_SCHEDULER;
  if (name == "simple") return SIMPLE_SCHEDULER;
  return DISCRETE_SCHEDULER;
}

// ─── Constructor (lightweight — just stores params, no model loading) ───

ImageContext::ImageContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ImageContext>(info) {
  auto env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Model path (string) required");
  }

  model_path_ = info[0].As<Napi::String>().Utf8Value();

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    clip_l_path_ = getStringOption(opts, "clipLPath", "");
    t5xxl_path_ = getStringOption(opts, "t5xxlPath", "");
    llm_path_ = getStringOption(opts, "llmPath", "");
    vae_path_ = getStringOption(opts, "vaePath", "");
    threads_ = getInt32Option(opts, "threads", -1);
    keep_vae_on_cpu_ = getBoolOption(opts, "keepVaeOnCpu", false);
    offload_to_cpu_ = getBoolOption(opts, "offloadToCpu", false);
    flash_attn_ = getBoolOption(opts, "flashAttn", true);

    if (opts.Has("vaeDecodeOnly") && opts.Get("vaeDecodeOnly").IsBoolean()) {
      vae_decode_only_ = opts.Get("vaeDecodeOnly").As<Napi::Boolean>().Value();
    }
  }

  // ctx_ remains nullptr — actual model loading happens async via InitWorker
}

// ─── InitWorker — loads model on background thread ───

class InitWorker : public Napi::AsyncWorker {
public:
  InitWorker(Napi::Env env, ImageContext* img_ctx, Napi::Object js_obj)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      img_ctx_(img_ctx),
      js_obj_ref_(Napi::Persistent(js_obj)),
      // Copy strings so they outlive the main-thread scope
      model_path_(img_ctx->model_path_),
      clip_l_path_(img_ctx->clip_l_path_),
      t5xxl_path_(img_ctx->t5xxl_path_),
      llm_path_(img_ctx->llm_path_),
      vae_path_(img_ctx->vae_path_),
      threads_(img_ctx->threads_),
      keep_vae_on_cpu_(img_ctx->keep_vae_on_cpu_),
      offload_to_cpu_(img_ctx->offload_to_cpu_),
      flash_attn_(img_ctx->flash_attn_),
      vae_decode_only_(img_ctx->vae_decode_only_) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);

    bool is_flux2 = !llm_path_.empty();
    bool is_flux1 = !clip_l_path_.empty() || !t5xxl_path_.empty();

    if (is_flux2) {
      params.diffusion_model_path = model_path_.c_str();
      params.llm_path = llm_path_.c_str();
      params.vae_path = vae_path_.empty() ? nullptr : vae_path_.c_str();
    } else if (is_flux1) {
      params.diffusion_model_path = model_path_.c_str();
      params.clip_l_path = clip_l_path_.empty() ? nullptr : clip_l_path_.c_str();
      params.t5xxl_path = t5xxl_path_.empty() ? nullptr : t5xxl_path_.c_str();
      params.vae_path = vae_path_.empty() ? nullptr : vae_path_.c_str();
    } else {
      params.model_path = model_path_.c_str();
      if (!vae_path_.empty()) {
        params.vae_path = vae_path_.c_str();
      }
    }

    params.vae_decode_only = vae_decode_only_;
    params.flash_attn = flash_attn_;
    params.diffusion_flash_attn = flash_attn_;
    params.keep_vae_on_cpu = keep_vae_on_cpu_;
    params.offload_params_to_cpu = offload_to_cpu_;
    params.n_threads = threads_ > 0 ? threads_ : sd_get_num_physical_cores();

    ctx_result_ = new_sd_ctx(&params);
    if (!ctx_result_) {
      SetError("Failed to create image generation context for: " + model_path_);
    }
  }

  void OnOK() override {
    // Set the loaded context on the ImageContext (main thread, safe)
    img_ctx_->ctx_ = ctx_result_;
    deferred_.Resolve(js_obj_ref_.Value());
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  ImageContext* img_ctx_;
  Napi::ObjectReference js_obj_ref_;
  sd_ctx_t* ctx_result_ = nullptr;

  // Copied params (string pointers in sd_ctx_params_t must remain valid during Execute)
  std::string model_path_;
  std::string clip_l_path_;
  std::string t5xxl_path_;
  std::string llm_path_;
  std::string vae_path_;
  int threads_;
  bool keep_vae_on_cpu_;
  bool offload_to_cpu_;
  bool flash_attn_;
  bool vae_decode_only_;
};

ImageContext::~ImageContext() {
  Cleanup();
}

void ImageContext::Cleanup() {
  if (ctx_) {
    free_sd_ctx(ctx_);
    ctx_ = nullptr;
  }
}

// ─── Generate ───

struct GenerateResult {
  std::vector<uint8_t> png_data;
  int width = 0;
  int height = 0;
};

class GenerateWorker : public Napi::AsyncWorker {
public:
  GenerateWorker(
    Napi::Env env,
    sd_ctx_t* ctx,
    std::string prompt,
    std::string negative_prompt,
    int width,
    int height,
    int steps,
    float cfg_scale,
    int64_t seed,
    sample_method_t sample_method,
    scheduler_t scheduler,
    int clip_skip,
    bool scheduler_specified
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      ctx_(ctx),
      prompt_(std::move(prompt)),
      negative_prompt_(std::move(negative_prompt)),
      width_(width), height_(height),
      steps_(steps), cfg_scale_(cfg_scale),
      seed_(seed), sample_method_(sample_method),
      scheduler_(scheduler), clip_skip_(clip_skip),
      scheduler_specified_(scheduler_specified) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    // Initialize generation params
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    gen_params.prompt = prompt_.c_str();
    gen_params.negative_prompt = negative_prompt_.c_str();
    gen_params.width = width_;
    gen_params.height = height_;
    gen_params.seed = seed_;
    gen_params.batch_count = 1;
    gen_params.clip_skip = clip_skip_;

    // CFG scale
    gen_params.sample_params.guidance.txt_cfg = cfg_scale_;

    // Sampling params
    gen_params.sample_params.sample_method = sample_method_;
    gen_params.sample_params.sample_steps = steps_;

    // Use model defaults for scheduler if not specified
    if (scheduler_specified_) {
      gen_params.sample_params.scheduler = scheduler_;
    } else {
      gen_params.sample_params.scheduler = sd_get_default_scheduler(ctx_, sample_method_);
    }

    // Generate
    sd_image_t* images = generate_image(ctx_, &gen_params);
    if (!images || !images[0].data) {
      SetError("Image generation failed");
      return;
    }

    // Encode to PNG
    result_.width = images[0].width;
    result_.height = images[0].height;

    PngBuffer png_buf;
    int ok = stbi_write_png_to_func(
      png_write_callback, &png_buf,
      images[0].width, images[0].height,
      images[0].channel,
      images[0].data,
      images[0].width * images[0].channel
    );

    // Free image data
    free(images[0].data);
    free(images);

    if (!ok) {
      SetError("Failed to encode PNG");
      return;
    }

    result_.png_data = std::move(png_buf.data);
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);

    auto buffer = Napi::Buffer<uint8_t>::Copy(
      env, result_.png_data.data(), result_.png_data.size()
    );

    deferred_.Resolve(buffer);
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  sd_ctx_t* ctx_;
  std::string prompt_;
  std::string negative_prompt_;
  int width_, height_, steps_;
  float cfg_scale_;
  int64_t seed_;
  sample_method_t sample_method_;
  scheduler_t scheduler_;
  int clip_skip_;
  bool scheduler_specified_;
  GenerateResult result_;
};

Napi::Value ImageContext::Generate(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!ctx_) {
    throw Napi::Error::New(env, "Image context not loaded");
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Prompt string required");
  }

  std::string prompt = info[0].As<Napi::String>().Utf8Value();

  // Defaults
  int width = 512;
  int height = 512;
  int steps = 20;
  float cfgScale = 7.0f;
  int64_t seed = -1; // random
  std::string negativePrompt = "";
  sample_method_t sampleMethod = EULER_SAMPLE_METHOD;
  scheduler_t scheduler = DISCRETE_SCHEDULER;
  int clipSkip = -1;
  bool schedulerSpecified = false;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    width = getInt32Option(opts, "width", 512);
    height = getInt32Option(opts, "height", 512);
    steps = getInt32Option(opts, "steps", 20);
    cfgScale = (float)getDoubleOption(opts, "cfgScale", 7.0);
    clipSkip = getInt32Option(opts, "clipSkip", -1);

    if (opts.Has("seed") && opts.Get("seed").IsNumber()) {
      seed = (int64_t)opts.Get("seed").As<Napi::Number>().Int64Value();
    }
    if (opts.Has("negativePrompt") && opts.Get("negativePrompt").IsString()) {
      negativePrompt = opts.Get("negativePrompt").As<Napi::String>().Utf8Value();
    }
    if (opts.Has("sampleMethod") && opts.Get("sampleMethod").IsString()) {
      sampleMethod = parse_sample_method(
        opts.Get("sampleMethod").As<Napi::String>().Utf8Value()
      );
    }
    if (opts.Has("scheduler") && opts.Get("scheduler").IsString()) {
      scheduler = parse_scheduler(
        opts.Get("scheduler").As<Napi::String>().Utf8Value()
      );
      schedulerSpecified = true;
    }
  }

  auto* worker = new GenerateWorker(
    env, ctx_,
    std::move(prompt), std::move(negativePrompt),
    width, height, steps, cfgScale, seed,
    sampleMethod, scheduler, clipSkip, schedulerSpecified
  );
  worker->Queue();
  return worker->Promise();
}

// ─── GenerateVideo ───

struct VideoResult {
  std::vector<std::vector<uint8_t>> frames; // PNG-encoded frames
  int width = 0;
  int height = 0;
};

class GenerateVideoWorker : public Napi::AsyncWorker {
public:
  GenerateVideoWorker(
    Napi::Env env,
    sd_ctx_t* ctx,
    std::string prompt,
    std::string negative_prompt,
    int width, int height,
    int video_frames,
    float cfg_scale,
    float flow_shift,
    int steps,
    int64_t seed,
    sample_method_t sample_method,
    scheduler_t scheduler,
    int clip_skip,
    bool scheduler_specified
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      ctx_(ctx),
      prompt_(std::move(prompt)),
      negative_prompt_(std::move(negative_prompt)),
      width_(width), height_(height),
      video_frames_(video_frames),
      cfg_scale_(cfg_scale),
      flow_shift_(flow_shift),
      steps_(steps), seed_(seed),
      sample_method_(sample_method),
      scheduler_(scheduler),
      clip_skip_(clip_skip),
      scheduler_specified_(scheduler_specified) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    sd_vid_gen_params_t vparams;
    sd_vid_gen_params_init(&vparams);

    vparams.prompt = prompt_.c_str();
    vparams.negative_prompt = negative_prompt_.c_str();
    vparams.width = width_;
    vparams.height = height_;
    vparams.video_frames = video_frames_;
    vparams.seed = seed_;
    vparams.clip_skip = clip_skip_;

    vparams.sample_params.guidance.txt_cfg = cfg_scale_;
    vparams.sample_params.sample_method = sample_method_;
    vparams.sample_params.sample_steps = steps_;
    vparams.sample_params.flow_shift = flow_shift_;

    if (scheduler_specified_) {
      vparams.sample_params.scheduler = scheduler_;
    } else {
      vparams.sample_params.scheduler = sd_get_default_scheduler(ctx_, sample_method_);
    }

    int num_frames_out = 0;
    sd_image_t* frames = generate_video(ctx_, &vparams, &num_frames_out);
    if (!frames || num_frames_out <= 0) {
      SetError("Video generation failed");
      return;
    }

    result_.width = frames[0].width;
    result_.height = frames[0].height;

    // Encode each frame as PNG
    for (int i = 0; i < num_frames_out; i++) {
      PngBuffer png_buf;
      int ok = stbi_write_png_to_func(
        png_write_callback, &png_buf,
        frames[i].width, frames[i].height,
        frames[i].channel, frames[i].data,
        frames[i].width * frames[i].channel
      );
      if (frames[i].data) free(frames[i].data);

      if (!ok) {
        // Free remaining frames
        for (int j = i + 1; j < num_frames_out; j++) {
          if (frames[j].data) free(frames[j].data);
        }
        free(frames);
        SetError("Failed to encode frame " + std::to_string(i) + " as PNG");
        return;
      }
      result_.frames.push_back(std::move(png_buf.data));
    }
    free(frames);
  }

  void OnOK() override {
    auto env = Env();
    Napi::HandleScope scope(env);

    // Return array of PNG buffers
    Napi::Array arr = Napi::Array::New(env, result_.frames.size());
    for (size_t i = 0; i < result_.frames.size(); i++) {
      auto buf = Napi::Buffer<uint8_t>::Copy(
        env, result_.frames[i].data(), result_.frames[i].size()
      );
      arr.Set((uint32_t)i, buf);
    }

    deferred_.Resolve(arr);
  }

  void OnError(const Napi::Error& e) override {
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  sd_ctx_t* ctx_;
  std::string prompt_;
  std::string negative_prompt_;
  int width_, height_, video_frames_;
  float cfg_scale_, flow_shift_;
  int steps_;
  int64_t seed_;
  sample_method_t sample_method_;
  scheduler_t scheduler_;
  int clip_skip_;
  bool scheduler_specified_;
  VideoResult result_;
};

Napi::Value ImageContext::GenerateVideo(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!ctx_) {
    throw Napi::Error::New(env, "Image context not loaded");
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Prompt string required");
  }

  std::string prompt = info[0].As<Napi::String>().Utf8Value();

  // Defaults for video generation
  int width = 832;
  int height = 480;
  int videoFrames = 33;
  int steps = 30;
  float cfgScale = 6.0f;
  float flowShift = 3.0f;
  int64_t seed = -1;
  std::string negativePrompt = "";
  sample_method_t sampleMethod = EULER_SAMPLE_METHOD;
  scheduler_t scheduler = DISCRETE_SCHEDULER;
  int clipSkip = -1;
  bool schedulerSpecified = false;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    width = getInt32Option(opts, "width", 832);
    height = getInt32Option(opts, "height", 480);
    videoFrames = getInt32Option(opts, "videoFrames", 33);
    steps = getInt32Option(opts, "steps", 30);
    cfgScale = (float)getDoubleOption(opts, "cfgScale", 6.0);
    flowShift = (float)getDoubleOption(opts, "flowShift", 3.0);
    clipSkip = getInt32Option(opts, "clipSkip", -1);

    if (opts.Has("seed") && opts.Get("seed").IsNumber()) {
      seed = (int64_t)opts.Get("seed").As<Napi::Number>().Int64Value();
    }
    if (opts.Has("negativePrompt") && opts.Get("negativePrompt").IsString()) {
      negativePrompt = opts.Get("negativePrompt").As<Napi::String>().Utf8Value();
    }
    if (opts.Has("sampleMethod") && opts.Get("sampleMethod").IsString()) {
      sampleMethod = parse_sample_method(
        opts.Get("sampleMethod").As<Napi::String>().Utf8Value()
      );
    }
    if (opts.Has("scheduler") && opts.Get("scheduler").IsString()) {
      scheduler = parse_scheduler(
        opts.Get("scheduler").As<Napi::String>().Utf8Value()
      );
      schedulerSpecified = true;
    }
  }

  auto* worker = new GenerateVideoWorker(
    env, ctx_,
    std::move(prompt), std::move(negativePrompt),
    width, height, videoFrames,
    cfgScale, flowShift, steps, seed,
    sampleMethod, scheduler, clipSkip, schedulerSpecified
  );
  worker->Queue();
  return worker->Promise();
}

// ─── Unload ───

Napi::Value ImageContext::Unload(const Napi::CallbackInfo& info) {
  Cleanup();
  auto deferred = Napi::Promise::Deferred::New(info.Env());
  deferred.Resolve(info.Env().Undefined());
  return deferred.Promise();
}

// ─── Module Factory (truly async — model loads on background thread) ───

Napi::Value CreateImageContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  Napi::Function constructor = env.GetInstanceData<Napi::FunctionReference>()->Value();

  std::vector<napi_value> args;
  for (size_t i = 0; i < info.Length(); i++) {
    args.push_back(info[i]);
  }

  // Constructor is now lightweight — just stores params
  Napi::Object ctx = constructor.New(args);

  // Launch async model loading on worker thread
  ImageContext* imgCtx = Napi::ObjectWrap<ImageContext>::Unwrap(ctx);
  auto* worker = new InitWorker(env, imgCtx, ctx);
  worker->Queue();
  return worker->Promise();
}

// ─── Module Init ───

Napi::Object InitImageModule(Napi::Env env, Napi::Object exports) {
  Napi::Function func = ImageContext::DefineClass(env, "ImageContext", {
    ImageContext::InstanceMethod("generate", &ImageContext::Generate),
    ImageContext::InstanceMethod("generateVideo", &ImageContext::GenerateVideo),
    ImageContext::InstanceMethod("unload", &ImageContext::Unload),
  });

  auto* constructor = new Napi::FunctionReference();
  *constructor = Napi::Persistent(func);
  constructor->SuppressDestruct();
  env.SetInstanceData(constructor);

  exports.Set("createContext", Napi::Function::New(env, CreateImageContext));
  return exports;
}

NODE_API_MODULE(image, InitImageModule)
