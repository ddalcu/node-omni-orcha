#pragma once
// Kokoro-82M TTS engine — ONNX Runtime inference with dictionary-based G2P.
// C API for N-API binding.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kokoro_context kokoro_context;

typedef struct kokoro_audio {
    float*  samples;       // PCM float32 mono (caller must free via kokoro_free_audio)
    int     n_samples;
    int     sample_rate;   // always 24000
} kokoro_audio;

// Create a Kokoro context. Loads the ONNX model, voice embeddings, and phoneme dictionary.
// model_path: path to kokoro-v1.0.fp16.onnx
// voices_path: path to voices.bin (converted from voices-v1.0.bin)
// dict_path: path to phoneme_dict.bin (generated from CMU dict)
// use_gpu: enable CoreML/CUDA acceleration
kokoro_context* kokoro_create(const char* model_path, const char* voices_path,
                              const char* dict_path, bool use_gpu);

// List available voice names. Caller must NOT free the returned array or strings.
// *count receives the number of voices.
const char** kokoro_list_voices(kokoro_context* ctx, int* count);

// Synthesize speech from text.
// voice: voice name (e.g. "af_heart"). NULL or "" for default.
// speed: speech rate (1.0 = normal, < 1 slower, > 1 faster).
// Returns allocated audio that must be freed with kokoro_free_audio.
kokoro_audio* kokoro_speak(kokoro_context* ctx, const char* text,
                           const char* voice, float speed);

// Free audio returned by kokoro_speak.
void kokoro_free_audio(kokoro_audio* audio);

// Destroy the context and free all resources.
void kokoro_free(kokoro_context* ctx);

// Get the last error message (or NULL if no error).
const char* kokoro_get_error(kokoro_context* ctx);

#ifdef __cplusplus
}
#endif
