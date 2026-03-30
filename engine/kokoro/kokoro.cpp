#include "kokoro.h"
#include "phonemize.h"
#include "vocab.h"
#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ─── Voice embedding file format ───
// Binary format (packed by download-kokoro-voices.ts):
//   uint32_t  num_voices
//   uint32_t  num_frames (510)
//   uint32_t  style_dim (256)
//   For each voice:
//     uint16_t  name_len
//     char[name_len]  name_utf8
//     float32[num_frames * style_dim]  embeddings
//
// When speaking, the style vector for a given token sequence of length L
// is voice.embeddings[min(L, num_frames-1)] — a float[style_dim] row.

static constexpr int STYLE_DIM = 256;
static constexpr int NUM_FRAMES = 510;
static constexpr int SAMPLE_RATE = 24000;
static constexpr int MAX_PHONEME_LENGTH = 509; // max index into voice frames

struct kokoro_context {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::MemoryInfo mem_info{nullptr};

    // Voice name → float[NUM_FRAMES * STYLE_DIM] embedding table
    std::unordered_map<std::string, std::vector<float>> voices;
    int num_frames = NUM_FRAMES;
    int style_dim = STYLE_DIM;
    std::vector<std::string> voice_names;
    std::vector<const char*> voice_name_ptrs;  // for C API return

    std::string default_voice;
    std::string last_error;
    std::mutex mutex;

    kokoro_context()
        : env(ORT_LOGGING_LEVEL_WARNING, "kokoro"),
          mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

// ─── Voice file reader ───

static bool load_voices(kokoro_context* ctx, const char* voices_path) {
    std::ifstream f(voices_path, std::ios::binary);
    if (!f.is_open()) {
        ctx->last_error = std::string("Cannot open voices file: ") + voices_path;
        return false;
    }

    // Read header: num_voices, num_frames, style_dim
    uint32_t num_voices = 0, nf = 0, sd = 0;
    f.read(reinterpret_cast<char*>(&num_voices), 4);
    f.read(reinterpret_cast<char*>(&nf), 4);
    f.read(reinterpret_cast<char*>(&sd), 4);
    if (!f || num_voices == 0 || num_voices > 1000 || nf == 0 || sd == 0) {
        ctx->last_error = "Invalid voices file header";
        return false;
    }
    ctx->num_frames = static_cast<int>(nf);
    ctx->style_dim = static_cast<int>(sd);
    size_t embed_size = static_cast<size_t>(nf) * sd;

    for (uint32_t i = 0; i < num_voices; i++) {
        uint16_t name_len = 0;
        f.read(reinterpret_cast<char*>(&name_len), 2);
        if (!f || name_len == 0 || name_len > 256) {
            ctx->last_error = "Invalid voice name length";
            return false;
        }

        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        std::vector<float> embedding(embed_size);
        f.read(reinterpret_cast<char*>(embedding.data()), embed_size * sizeof(float));

        if (!f) {
            ctx->last_error = std::string("Truncated voice data for: ") + name;
            return false;
        }

        ctx->voice_names.push_back(name);
        ctx->voices[name] = std::move(embedding);
    }

    // Build C string pointer array for kokoro_list_voices
    ctx->voice_name_ptrs.clear();
    ctx->voice_name_ptrs.reserve(ctx->voice_names.size());
    for (const auto& name : ctx->voice_names) {
        ctx->voice_name_ptrs.push_back(name.c_str());
    }

    // Default voice: af_heart if available, otherwise first
    if (ctx->voices.count("af_heart")) {
        ctx->default_voice = "af_heart";
    } else if (!ctx->voice_names.empty()) {
        ctx->default_voice = ctx->voice_names[0];
    }

    return true;
}

// ─── C API ───

kokoro_context* kokoro_create(const char* model_path, const char* voices_path,
                              const char* dict_path, bool use_gpu) {
    // Initialize dictionary-based phonemizer
    if (!kokoro::g2p_init(dict_path)) {
        fprintf(stderr, "[kokoro] Failed to load phoneme dictionary: %s\n", dict_path);
        return nullptr;
    }

    auto* ctx = new (std::nothrow) kokoro_context();
    if (!ctx) return nullptr;

    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (use_gpu) {
#ifdef __APPLE__
            // CoreML execution provider on macOS
            try {
                uint32_t coreml_flags = 0;
                coreml_flags |= 0x004; // COREML_FLAG_ENABLE_ON_SUBGRAPH
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coreml_flags));
                fprintf(stderr, "[kokoro] CoreML execution provider enabled\n");
            } catch (const Ort::Exception& e) {
                fprintf(stderr, "[kokoro] CoreML not available, using CPU: %s\n", e.what());
            }
#elif defined(OMNI_GPU_CUDA)
            // CUDA execution provider on Linux/Windows
            try {
                OrtCUDAProviderOptions cuda_opts{};
                cuda_opts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                fprintf(stderr, "[kokoro] CUDA execution provider enabled\n");
            } catch (const Ort::Exception& e) {
                fprintf(stderr, "[kokoro] CUDA not available, using CPU: %s\n", e.what());
            }
#endif
        }

#ifdef _WIN32
        // MSVC Ort::Session requires wide string path
        int wlen = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, nullptr, 0);
        std::wstring wpath(wlen, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wpath.data(), wlen);
        ctx->session = Ort::Session(ctx->env, wpath.c_str(), opts);
#else
        ctx->session = Ort::Session(ctx->env, model_path, opts);
#endif
    } catch (const Ort::Exception& e) {
        ctx->last_error = std::string("ONNX session creation failed: ") + e.what();
        delete ctx;
        return nullptr;
    }

    // Load voice embeddings
    if (!load_voices(ctx, voices_path)) {
        delete ctx;
        return nullptr;
    }

    fprintf(stderr, "[kokoro] Loaded %zu voices, default: %s\n",
            ctx->voices.size(), ctx->default_voice.c_str());
    return ctx;
}

const char** kokoro_list_voices(kokoro_context* ctx, int* count) {
    if (!ctx) { if (count) *count = 0; return nullptr; }
    if (count) *count = static_cast<int>(ctx->voice_name_ptrs.size());
    return ctx->voice_name_ptrs.data();
}

kokoro_audio* kokoro_speak(kokoro_context* ctx, const char* text,
                           const char* voice, float speed) {
    if (!ctx || !text || !*text) return nullptr;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->last_error.clear();

    // Resolve voice
    std::string voice_name = (voice && *voice) ? voice : ctx->default_voice;
    auto vit = ctx->voices.find(voice_name);
    if (vit == ctx->voices.end()) {
        ctx->last_error = std::string("Unknown voice: ") + voice_name;
        return nullptr;
    }
    const std::vector<float>& voice_data = vit->second; // shape: [num_frames, style_dim]

    // Speed bounds
    if (speed <= 0.0f) speed = 1.0f;
    if (speed > 5.0f) speed = 5.0f;

    // Text → phonemes → token IDs (using dictionary-based G2P)
    auto token_ids = kokoro::text_to_token_ids(text);
    if (token_ids.size() <= 2) {
        ctx->last_error = "Phonemization produced no tokens for the given text";
        return nullptr;
    }

    // Truncate if too long (model max ~510 tokens excluding pads)
    if (token_ids.size() > MAX_PHONEME_LENGTH + 2) {
        token_ids.resize(MAX_PHONEME_LENGTH + 1);
        token_ids.push_back(0); // re-add trailing pad
    }

    try {
        // Prepare input tensors
        int64_t seq_len = static_cast<int64_t>(token_ids.size());

        // input_ids: int64[1, seq_len]
        std::array<int64_t, 2> ids_shape = {1, seq_len};
        auto ids_tensor = Ort::Value::CreateTensor<int64_t>(
            ctx->mem_info, token_ids.data(), token_ids.size(),
            ids_shape.data(), ids_shape.size());

        // style: float32[1, style_dim]
        // Index by phoneme count WITHOUT pad tokens (matches Python reference:
        //   voice = voice[len(tokens)]   ← before pads
        //   tokens = [[0, *tokens, 0]]   ← pads added after
        int phoneme_count = static_cast<int>(token_ids.size()) - 2; // exclude leading+trailing pad
        if (phoneme_count < 0) phoneme_count = 0;
        int frame_idx = std::min(phoneme_count, ctx->num_frames - 1);
        std::vector<float> style_buf(
            voice_data.begin() + frame_idx * ctx->style_dim,
            voice_data.begin() + frame_idx * ctx->style_dim + ctx->style_dim);
        std::array<int64_t, 2> style_shape = {1, static_cast<int64_t>(ctx->style_dim)};
        auto style_tensor = Ort::Value::CreateTensor<float>(
            ctx->mem_info, style_buf.data(), style_buf.size(),
            style_shape.data(), style_shape.size());

        // speed: float32 scalar [1]
        std::array<float, 1> speed_buf = {speed};
        std::array<int64_t, 1> speed_shape = {1};
        auto speed_tensor = Ort::Value::CreateTensor<float>(
            ctx->mem_info, speed_buf.data(), speed_buf.size(),
            speed_shape.data(), speed_shape.size());

        // Run inference
        const char* input_names[] = {"input_ids", "style", "speed"};
        const char* output_names[] = {"waveform"};
        Ort::Value inputs[] = {std::move(ids_tensor), std::move(style_tensor), std::move(speed_tensor)};

        auto outputs = ctx->session.Run(
            Ort::RunOptions{nullptr},
            input_names, inputs, 3,
            output_names, 1);

        // Extract output audio
        auto& output = outputs[0];
        auto type_info = output.GetTensorTypeAndShapeInfo();
        auto shape = type_info.GetShape();
        size_t n_samples = 1;
        for (auto dim : shape) {
            if (dim > 0) n_samples *= static_cast<size_t>(dim);
        }

        const float* data = output.GetTensorData<float>();

        // Allocate result
        auto* audio = new kokoro_audio();
        audio->samples = new float[n_samples];
        audio->n_samples = static_cast<int>(n_samples);
        audio->sample_rate = SAMPLE_RATE;
        std::memcpy(audio->samples, data, n_samples * sizeof(float));

        return audio;
    } catch (const Ort::Exception& e) {
        ctx->last_error = std::string("ONNX inference failed: ") + e.what();
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("Inference error: ") + e.what();
        return nullptr;
    }
}

void kokoro_free_audio(kokoro_audio* audio) {
    if (audio) {
        delete[] audio->samples;
        delete audio;
    }
}

void kokoro_free(kokoro_context* ctx) {
    if (ctx) {
        delete ctx;
    }
}

const char* kokoro_get_error(kokoro_context* ctx) {
    if (!ctx || ctx->last_error.empty()) return nullptr;
    return ctx->last_error.c_str();
}
