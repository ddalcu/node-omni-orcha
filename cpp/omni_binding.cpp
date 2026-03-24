#include "llm_context.h"
#include "stt_context.h"
#include "tts_context.h"
#include "image_context.h"
#include <napi.h>

Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
  LlmContext::Register(env, exports);
  SttContext::Register(env, exports);
  TtsContext::Register(env, exports);
  ImageContext::Register(env, exports);

  return exports;
}

NODE_API_MODULE(omni, InitModule)
