#include "llm_context.h"
#include "stt_context.h"
#include "tts_context.h"
#include "image_context.h"
#ifdef KOKORO_ENABLED
#include "kokoro_context.h"
#endif
#include <napi.h>

Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
  LlmContext::Register(env, exports);
  SttContext::Register(env, exports);
  TtsContext::Register(env, exports);
  ImageContext::Register(env, exports);
#ifdef KOKORO_ENABLED
  KokoroContext::Register(env, exports);
  exports.Set("kokoroEnabled", Napi::Boolean::New(env, true));
#else
  exports.Set("kokoroEnabled", Napi::Boolean::New(env, false));
#endif

  return exports;
}

NODE_API_MODULE(omni, InitModule)
