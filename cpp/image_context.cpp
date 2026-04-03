#include "image_context.h"
#include "common/error_helpers.h"
#include "stable-diffusion.h"

// stb_image for decoding init images (PNG/JPEG buffers)
// STB_IMAGE_IMPLEMENTATION is defined in mtmd-helper.cpp — only include the header here
#include "stb_image.h"

// stb_image_write for PNG encoding
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <string>
#include <vector>
#include <cstring>

Napi::FunctionReference ImageContext::constructor_;

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

// ─── Constructor ───

ImageContext::ImageContext(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ImageContext>(info) {
  auto env = info.Env();

  if (info.Length() < 1 || !info[0].IsString()) {
    throw Napi::TypeError::New(env, "Model path (string) required");
  }

  model_path_ = info[0].As<Napi::String>().Utf8Value();
  if (model_path_.empty()) {
    throw Napi::TypeError::New(env, "Model path must not be empty");
  }

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    clip_l_path_ = getStringOption(opts, "clipLPath", "");
    t5xxl_path_ = getStringOption(opts, "t5xxlPath", "");
    llm_path_ = getStringOption(opts, "llmPath", "");
    vae_path_ = getStringOption(opts, "vaePath", "");
    high_noise_diffusion_model_path_ = getStringOption(opts, "highNoiseDiffusionModelPath", "");
    threads_ = getInt32Option(opts, "threads", -1);
    keep_vae_on_cpu_ = getBoolOption(opts, "keepVaeOnCpu", false);
    offload_to_cpu_ = getBoolOption(opts, "offloadToCpu", false);
    flash_attn_ = getBoolOption(opts, "flashAttn", true);

    if (opts.Has("vaeDecodeOnly") && opts.Get("vaeDecodeOnly").IsBoolean()) {
      vae_decode_only_ = opts.Get("vaeDecodeOnly").As<Napi::Boolean>().Value();
    }

    // Video models (WAN) use t5xxl text encoder, while FLUX image models use llm.
    // SD models use neither. Only t5xxl-based models support generateVideo.
    is_video_model_ = !t5xxl_path_.empty() && llm_path_.empty();
  }
}

// ─── InitWorker ───

class InitWorker : public Napi::AsyncWorker {
public:
  InitWorker(Napi::Env env, ImageContext* img_ctx, Napi::Object js_obj)
    : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      img_ctx_(img_ctx),
      js_obj_ref_(Napi::Persistent(js_obj)),
      model_path_(img_ctx->model_path_),
      clip_l_path_(img_ctx->clip_l_path_),
      t5xxl_path_(img_ctx->t5xxl_path_),
      llm_path_(img_ctx->llm_path_),
      vae_path_(img_ctx->vae_path_),
      high_noise_diffusion_model_path_(img_ctx->high_noise_diffusion_model_path_),
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
    params.free_params_immediately = false;

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

    if (!high_noise_diffusion_model_path_.empty()) {
      params.high_noise_diffusion_model_path = high_noise_diffusion_model_path_.c_str();
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
  std::string model_path_;
  std::string clip_l_path_;
  std::string t5xxl_path_;
  std::string llm_path_;
  std::string vae_path_;
  std::string high_noise_diffusion_model_path_;
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
    int width, int height, int steps,
    float cfg_scale, int64_t seed,
    sample_method_t sample_method,
    scheduler_t scheduler, int clip_skip,
    bool scheduler_specified,
    std::atomic<bool>& busy_flag
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      ctx_(ctx),
      prompt_(std::move(prompt)),
      negative_prompt_(std::move(negative_prompt)),
      width_(width), height_(height),
      steps_(steps), cfg_scale_(cfg_scale),
      seed_(seed), sample_method_(sample_method),
      scheduler_(scheduler), clip_skip_(clip_skip),
      scheduler_specified_(scheduler_specified),
      busy_flag_(busy_flag) {}

  Napi::Promise Promise() { return deferred_.Promise(); }

protected:
  void Execute() override {
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    gen_params.prompt = prompt_.c_str();
    gen_params.negative_prompt = negative_prompt_.c_str();
    gen_params.width = width_;
    gen_params.height = height_;
    gen_params.seed = seed_;
    gen_params.batch_count = 1;
    gen_params.clip_skip = clip_skip_;
    gen_params.sample_params.guidance.txt_cfg = cfg_scale_;
    gen_params.sample_params.sample_method = sample_method_;
    gen_params.sample_params.sample_steps = steps_;

    if (scheduler_specified_) {
      gen_params.sample_params.scheduler = scheduler_;
    } else {
      gen_params.sample_params.scheduler = sd_get_default_scheduler(ctx_, sample_method_);
    }

    sd_image_t* images = generate_image(ctx_, &gen_params);
    if (!images || !images[0].data) {
      SetError("Image generation failed");
      return;
    }

    result_.width = images[0].width;
    result_.height = images[0].height;

    PngBuffer png_buf;
    int ok = stbi_write_png_to_func(
      png_write_callback, &png_buf,
      images[0].width, images[0].height,
      images[0].channel, images[0].data,
      images[0].width * images[0].channel
    );

    free(images[0].data);
    free(images);

    if (!ok) {
      SetError("Failed to encode PNG");
      return;
    }

    result_.png_data = std::move(png_buf.data);
  }

  void OnOK() override {
    busy_flag_.store(false);
    auto env = Env();
    Napi::HandleScope scope(env);
    auto buffer = Napi::Buffer<uint8_t>::Copy(
      env, result_.png_data.data(), result_.png_data.size()
    );
    deferred_.Resolve(buffer);
  }

  void OnError(const Napi::Error& e) override {
    busy_flag_.store(false);
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
  std::atomic<bool>& busy_flag_;
  GenerateResult result_;
};

Napi::Value ImageContext::Generate(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!ctx_) throw Napi::Error::New(env, "Image context not loaded");
  if (!AcquireBusy()) {
    throw Napi::Error::New(env, "Context is busy — another inference operation is in progress. "
      "Wait for the previous call to complete before starting a new one.");
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    ReleaseBusy();
    throw Napi::TypeError::New(env, "Prompt string required");
  }

  std::string prompt = info[0].As<Napi::String>().Utf8Value();

  int width = 512, height = 512, steps = 20;
  float cfgScale = 7.0f;
  int64_t seed = -1;
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

    if (opts.Has("seed") && opts.Get("seed").IsNumber())
      seed = (int64_t)opts.Get("seed").As<Napi::Number>().Int64Value();
    if (opts.Has("negativePrompt") && opts.Get("negativePrompt").IsString())
      negativePrompt = opts.Get("negativePrompt").As<Napi::String>().Utf8Value();
    if (opts.Has("sampleMethod") && opts.Get("sampleMethod").IsString())
      sampleMethod = parse_sample_method(opts.Get("sampleMethod").As<Napi::String>().Utf8Value());
    if (opts.Has("scheduler") && opts.Get("scheduler").IsString()) {
      scheduler = parse_scheduler(opts.Get("scheduler").As<Napi::String>().Utf8Value());
      schedulerSpecified = true;
    }
  }

  if (width < 1 || height < 1) {
    ReleaseBusy();
    throw Napi::Error::New(env, "Width and height must be at least 1");
  }
  if (steps < 1) {
    ReleaseBusy();
    throw Napi::Error::New(env, "Steps must be at least 1");
  }

  auto* worker = new GenerateWorker(
    env, ctx_, std::move(prompt), std::move(negativePrompt),
    width, height, steps, cfgScale, seed,
    sampleMethod, scheduler, clipSkip, schedulerSpecified, busy_
  );
  worker->Queue();
  return worker->Promise();
}

// ─── GenerateVideo ───

struct VideoResult {
  std::vector<std::vector<uint8_t>> frames;
  int width = 0;
  int height = 0;
};

class GenerateVideoWorker : public Napi::AsyncWorker {
public:
  GenerateVideoWorker(
    Napi::Env env, sd_ctx_t* ctx,
    std::string prompt, std::string negative_prompt,
    int width, int height, int video_frames,
    float cfg_scale, float flow_shift, int steps, int64_t seed,
    sample_method_t sample_method, scheduler_t scheduler,
    int clip_skip, bool scheduler_specified,
    int high_noise_steps, float high_noise_cfg_scale,
    sample_method_t high_noise_sample_method, bool high_noise_specified,
    std::vector<uint8_t> init_image_data, std::vector<uint8_t> end_image_data,
    std::atomic<bool>& busy_flag
  ) : Napi::AsyncWorker(env),
      deferred_(Napi::Promise::Deferred::New(env)),
      ctx_(ctx), prompt_(std::move(prompt)),
      negative_prompt_(std::move(negative_prompt)),
      width_(width), height_(height), video_frames_(video_frames),
      cfg_scale_(cfg_scale), flow_shift_(flow_shift),
      steps_(steps), seed_(seed),
      sample_method_(sample_method), scheduler_(scheduler),
      clip_skip_(clip_skip), scheduler_specified_(scheduler_specified),
      high_noise_steps_(high_noise_steps),
      high_noise_cfg_scale_(high_noise_cfg_scale),
      high_noise_sample_method_(high_noise_sample_method),
      high_noise_specified_(high_noise_specified),
      init_image_data_(std::move(init_image_data)),
      end_image_data_(std::move(end_image_data)),
      busy_flag_(busy_flag) {}

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

    if (high_noise_specified_) {
      vparams.high_noise_sample_params.sample_steps = high_noise_steps_;
      vparams.high_noise_sample_params.guidance.txt_cfg = high_noise_cfg_scale_;
      vparams.high_noise_sample_params.sample_method = high_noise_sample_method_;
    }

    // Decode init image (I2V / TI2V — first frame)
    int init_w = 0, init_h = 0, init_c = 0;
    uint8_t* init_pixels = nullptr;
    if (!init_image_data_.empty()) {
      init_pixels = stbi_load_from_memory(
        init_image_data_.data(), (int)init_image_data_.size(),
        &init_w, &init_h, &init_c, 3);
      if (init_pixels) {
        vparams.init_image.width = init_w;
        vparams.init_image.height = init_h;
        vparams.init_image.channel = 3;
        vparams.init_image.data = init_pixels;
      }
    }

    // Decode end image (FLF2V — last frame)
    int end_w = 0, end_h = 0, end_c = 0;
    uint8_t* end_pixels = nullptr;
    if (!end_image_data_.empty()) {
      end_pixels = stbi_load_from_memory(
        end_image_data_.data(), (int)end_image_data_.size(),
        &end_w, &end_h, &end_c, 3);
      if (end_pixels) {
        vparams.end_image.width = end_w;
        vparams.end_image.height = end_h;
        vparams.end_image.channel = 3;
        vparams.end_image.data = end_pixels;
      }
    }

    int num_frames_out = 0;
    sd_image_t* frames = generate_video(ctx_, &vparams, &num_frames_out);

    if (init_pixels) stbi_image_free(init_pixels);
    if (end_pixels) stbi_image_free(end_pixels);
    if (!frames || num_frames_out <= 0) {
      SetError("Video generation failed");
      return;
    }

    result_.width = frames[0].width;
    result_.height = frames[0].height;

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
    busy_flag_.store(false);
    auto env = Env();
    Napi::HandleScope scope(env);
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
    busy_flag_.store(false);
    deferred_.Reject(e.Value());
  }

private:
  Napi::Promise::Deferred deferred_;
  sd_ctx_t* ctx_;
  std::string prompt_, negative_prompt_;
  int width_, height_, video_frames_;
  float cfg_scale_, flow_shift_;
  int steps_;
  int64_t seed_;
  sample_method_t sample_method_;
  scheduler_t scheduler_;
  int clip_skip_;
  bool scheduler_specified_;
  int high_noise_steps_;
  float high_noise_cfg_scale_;
  sample_method_t high_noise_sample_method_;
  bool high_noise_specified_;
  std::atomic<bool>& busy_flag_;
  std::vector<uint8_t> init_image_data_;
  std::vector<uint8_t> end_image_data_;
  VideoResult result_;
};

Napi::Value ImageContext::GenerateVideo(const Napi::CallbackInfo& info) {
  auto env = info.Env();

  if (!ctx_) throw Napi::Error::New(env, "Image context not loaded");
  if (!AcquireBusy()) {
    throw Napi::Error::New(env, "Context is busy — another inference operation is in progress. "
      "Wait for the previous call to complete before starting a new one.");
  }
  if (!is_video_model_) {
    ReleaseBusy();
    throw Napi::Error::New(env, "This model does not support video generation. Use a WAN/video model with t5xxlPath.");
  }
  if (info.Length() < 1 || !info[0].IsString()) {
    ReleaseBusy();
    throw Napi::TypeError::New(env, "Prompt string required");
  }

  std::string prompt = info[0].As<Napi::String>().Utf8Value();

  int width = 832, height = 480, videoFrames = 33, steps = 30;
  float cfgScale = 6.0f, flowShift = 3.0f;
  int64_t seed = -1;
  std::string negativePrompt = "";
  sample_method_t sampleMethod = EULER_SAMPLE_METHOD;
  scheduler_t scheduler = DISCRETE_SCHEDULER;
  int clipSkip = -1;
  bool schedulerSpecified = false;
  int highNoiseSteps = -1;
  float highNoiseCfgScale = -1.0f;
  sample_method_t highNoiseSampleMethod = EULER_SAMPLE_METHOD;
  bool highNoiseSpecified = false;

  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    width = getInt32Option(opts, "width", 832);
    height = getInt32Option(opts, "height", 480);
    videoFrames = getInt32Option(opts, "videoFrames", 33);
    steps = getInt32Option(opts, "steps", 30);
    cfgScale = (float)getDoubleOption(opts, "cfgScale", 6.0);
    flowShift = (float)getDoubleOption(opts, "flowShift", 3.0);
    clipSkip = getInt32Option(opts, "clipSkip", -1);

    if (opts.Has("seed") && opts.Get("seed").IsNumber())
      seed = (int64_t)opts.Get("seed").As<Napi::Number>().Int64Value();
    if (opts.Has("negativePrompt") && opts.Get("negativePrompt").IsString())
      negativePrompt = opts.Get("negativePrompt").As<Napi::String>().Utf8Value();
    if (opts.Has("sampleMethod") && opts.Get("sampleMethod").IsString())
      sampleMethod = parse_sample_method(opts.Get("sampleMethod").As<Napi::String>().Utf8Value());
    if (opts.Has("scheduler") && opts.Get("scheduler").IsString()) {
      scheduler = parse_scheduler(opts.Get("scheduler").As<Napi::String>().Utf8Value());
      schedulerSpecified = true;
    }
    if (opts.Has("highNoiseSteps") && opts.Get("highNoiseSteps").IsNumber()) {
      highNoiseSteps = opts.Get("highNoiseSteps").As<Napi::Number>().Int32Value();
      highNoiseSpecified = true;
    }
    if (opts.Has("highNoiseCfgScale") && opts.Get("highNoiseCfgScale").IsNumber()) {
      highNoiseCfgScale = (float)opts.Get("highNoiseCfgScale").As<Napi::Number>().DoubleValue();
      highNoiseSpecified = true;
    }
    if (opts.Has("highNoiseSampleMethod") && opts.Get("highNoiseSampleMethod").IsString()) {
      highNoiseSampleMethod = parse_sample_method(
        opts.Get("highNoiseSampleMethod").As<Napi::String>().Utf8Value()
      );
      highNoiseSpecified = true;
    }
  }

  if (width < 1 || height < 1) {
    ReleaseBusy();
    throw Napi::Error::New(env, "Width and height must be at least 1");
  }
  if (videoFrames < 1) {
    ReleaseBusy();
    throw Napi::Error::New(env, "videoFrames must be at least 1");
  }
  if (steps < 1) {
    ReleaseBusy();
    throw Napi::Error::New(env, "Steps must be at least 1");
  }

  // Extract init/end image buffers for I2V / TI2V / FLF2V
  std::vector<uint8_t> initImageData, endImageData;
  if (info.Length() >= 2 && info[1].IsObject()) {
    Napi::Object opts = info[1].As<Napi::Object>();
    if (opts.Has("initImage") && opts.Get("initImage").IsBuffer()) {
      auto buf = opts.Get("initImage").As<Napi::Buffer<uint8_t>>();
      initImageData.assign(buf.Data(), buf.Data() + buf.Length());
    }
    if (opts.Has("endImage") && opts.Get("endImage").IsBuffer()) {
      auto buf = opts.Get("endImage").As<Napi::Buffer<uint8_t>>();
      endImageData.assign(buf.Data(), buf.Data() + buf.Length());
    }
  }

  auto* worker = new GenerateVideoWorker(
    env, ctx_, std::move(prompt), std::move(negativePrompt),
    width, height, videoFrames, cfgScale, flowShift, steps, seed,
    sampleMethod, scheduler, clipSkip, schedulerSpecified,
    highNoiseSteps, highNoiseCfgScale, highNoiseSampleMethod, highNoiseSpecified,
    std::move(initImageData), std::move(endImageData), busy_
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

// ─── Registration ───

static Napi::Value CreateImageContext(const Napi::CallbackInfo& info) {
  auto env = info.Env();
  std::vector<napi_value> args;
  for (size_t i = 0; i < info.Length(); i++) args.push_back(info[i]);

  Napi::Object ctx = ImageContext::constructor_.New(args);
  ImageContext* imgCtx = Napi::ObjectWrap<ImageContext>::Unwrap(ctx);
  auto* worker = new InitWorker(env, imgCtx, ctx);
  worker->Queue();
  return worker->Promise();
}

void ImageContext::Register(Napi::Env env, Napi::Object& exports) {
  Napi::Function func = DefineClass(env, "ImageContext", {
    InstanceMethod("generate", &ImageContext::Generate),
    InstanceMethod("generateVideo", &ImageContext::GenerateVideo),
    InstanceMethod("unload", &ImageContext::Unload),
  });

  constructor_ = Napi::Persistent(func);
  constructor_.SuppressDestruct();

  exports.Set("createImageContext", Napi::Function::New(env, CreateImageContext));
}
