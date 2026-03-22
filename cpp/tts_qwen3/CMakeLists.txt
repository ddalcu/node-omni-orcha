cmake_minimum_required(VERSION 3.15)

execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

set(QWEN3_TTS_DIR "${CMAKE_SOURCE_DIR}/deps/qwen3-tts.cpp")

add_library(tts_qwen3_addon SHARED
  tts_qwen3_binding.cpp
  # Include the C API implementation directly
  "${QWEN3_TTS_DIR}/src/qwen3tts_c_api.cpp"
  ${CMAKE_JS_SRC}
)

set_target_properties(tts_qwen3_addon PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  OUTPUT_NAME "tts_qwen3"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)

target_include_directories(tts_qwen3_addon PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${CMAKE_SOURCE_DIR}/cpp/common
  "${QWEN3_TTS_DIR}/src"
  "${QWEN3_TTS_DIR}/ggml/include"
)

# Link against pre-built static libraries
target_link_libraries(tts_qwen3_addon PRIVATE
  ${CMAKE_JS_LIB}
  "${QWEN3_TTS_DIR}/build/libqwen3_tts.a"
  "${QWEN3_TTS_DIR}/build/libtext_tokenizer.a"
  "${QWEN3_TTS_DIR}/build/libtts_transformer.a"
  "${QWEN3_TTS_DIR}/build/libaudio_tokenizer_encoder.a"
  "${QWEN3_TTS_DIR}/build/libaudio_tokenizer_decoder.a"
  "${QWEN3_TTS_DIR}/ggml/build/src/libggml.a"
  "${QWEN3_TTS_DIR}/ggml/build/src/libggml-cpu.a"
  "${QWEN3_TTS_DIR}/ggml/build/src/ggml-metal/libggml-metal.a"
  "${QWEN3_TTS_DIR}/ggml/build/src/ggml-blas/libggml-blas.a"
  "${QWEN3_TTS_DIR}/ggml/build/src/libggml-base.a"
)

if(APPLE)
  target_link_libraries(tts_qwen3_addon PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework Accelerate"
  )
endif()

target_compile_definitions(tts_qwen3_addon PRIVATE
  NAPI_VERSION=9
  NODE_ADDON_API_DISABLE_DEPRECATED
)
