cmake_minimum_required(VERSION 3.15)

execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

set(QWEN3_TTS_DIR "${CMAKE_SOURCE_DIR}/deps/qwen3-tts.cpp")
set(QWEN3_BUILD_DIR "${QWEN3_TTS_DIR}/build")
set(Q3_GGML_BUILD "${QWEN3_TTS_DIR}/ggml/build")

# Platform-aware library paths (Unix: lib*.a, MSVC: *.lib in Release/)
set(P "${CMAKE_STATIC_LIBRARY_PREFIX}")
set(S "${CMAKE_STATIC_LIBRARY_SUFFIX}")
if(MSVC)
  set(CFG "/Release")
else()
  set(CFG "")
endif()

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

# Link against pre-built static libraries (platform-aware paths)
target_link_libraries(tts_qwen3_addon PRIVATE
  ${CMAKE_JS_LIB}
  "${QWEN3_BUILD_DIR}${CFG}/${P}qwen3_tts${S}"
  "${QWEN3_BUILD_DIR}${CFG}/${P}text_tokenizer${S}"
  "${QWEN3_BUILD_DIR}${CFG}/${P}tts_transformer${S}"
  "${QWEN3_BUILD_DIR}${CFG}/${P}audio_tokenizer_encoder${S}"
  "${QWEN3_BUILD_DIR}${CFG}/${P}audio_tokenizer_decoder${S}"
  "${Q3_GGML_BUILD}/src${CFG}/${P}ggml${S}"
  "${Q3_GGML_BUILD}/src${CFG}/${P}ggml-cpu${S}"
  "${Q3_GGML_BUILD}/src${CFG}/${P}ggml-base${S}"
)

if(APPLE)
  target_link_libraries(tts_qwen3_addon PRIVATE
    "${Q3_GGML_BUILD}/src/ggml-metal${CFG}/${P}ggml-metal${S}"
    "${Q3_GGML_BUILD}/src/ggml-blas${CFG}/${P}ggml-blas${S}"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework Accelerate"
  )
endif()

if(GPU_BACKEND STREQUAL "cuda")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(tts_qwen3_addon PRIVATE
    "${Q3_GGML_BUILD}/src/ggml-cuda${CFG}/${P}ggml-cuda${S}"
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
  )
endif()

target_compile_definitions(tts_qwen3_addon PRIVATE
  NAPI_VERSION=9
  NODE_ADDON_API_DISABLE_DEPRECATED
)
