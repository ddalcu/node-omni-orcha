cmake_minimum_required(VERSION 3.15)
project(node_omni_orcha)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# GPU backend selection: metal, cuda, cpu, auto
if(NOT DEFINED GPU_BACKEND)
  set(GPU_BACKEND "auto")
endif()

# Auto-detect GPU backend
if(GPU_BACKEND STREQUAL "auto")
  if(APPLE)
    set(GPU_BACKEND "metal")
  else()
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(GPU_BACKEND "cuda")
    else()
      set(GPU_BACKEND "cpu")
    endif()
  endif()
  message(STATUS "[omni] Auto-detected GPU backend: ${GPU_BACKEND}")
endif()

# Build static libraries
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# --- Engine (llama.cpp fork) ---
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

# Disable BLAS to avoid needing ggml-blas backend
set(GGML_BLAS OFF CACHE BOOL "" FORCE)

# sd.cpp requires GGML_MAX_NAME >= 128
add_compile_definitions(GGML_MAX_NAME=128)

if(GPU_BACKEND STREQUAL "metal")
  set(GGML_METAL ON CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
elseif(GPU_BACKEND STREQUAL "cuda")
  set(GGML_METAL OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
else()
  set(GGML_METAL OFF CACHE BOOL "" FORCE)
  set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(engine)
message(STATUS "[omni] Engine configured with GPU_BACKEND=${GPU_BACKEND}")

# --- Find node-addon-api ---
execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

# --- Whisper (STT engine) ---
# Compile whisper.cpp directly against our shared ggml
add_library(whisper STATIC
  engine/stt/whisper.cpp
)
target_include_directories(whisper PUBLIC
  ${CMAKE_SOURCE_DIR}/engine/stt
)
target_include_directories(whisper PRIVATE
  ${CMAKE_SOURCE_DIR}/engine/ggml/include
)
target_link_libraries(whisper PRIVATE ggml)
target_compile_features(whisper PRIVATE cxx_std_17)
target_compile_definitions(whisper PRIVATE WHISPER_VERSION="1.0.0")
set_target_properties(whisper PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Qwen3-TTS (TTS engine) ---
# Compile qwen3-tts sources directly against our shared ggml
add_library(qwen3_tts STATIC
  engine/tts/text_tokenizer.cpp
  engine/tts/tts_transformer.cpp
  engine/tts/audio_tokenizer_encoder.cpp
  engine/tts/audio_tokenizer_decoder.cpp
  engine/tts/gguf_loader.cpp
  engine/tts/qwen3_tts.cpp
  engine/tts/qwen3tts_c_api.cpp
  engine/tts/coreml_code_predictor_stub.cpp
)
target_include_directories(qwen3_tts PUBLIC
  ${CMAKE_SOURCE_DIR}/engine/tts
)
target_include_directories(qwen3_tts PRIVATE
  ${CMAKE_SOURCE_DIR}/engine/ggml/include
  ${CMAKE_SOURCE_DIR}/engine/ggml/src
)
target_link_libraries(qwen3_tts PRIVATE ggml)
target_compile_features(qwen3_tts PRIVATE cxx_std_17)
set_target_properties(qwen3_tts PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Stable Diffusion (Image/Video engine) ---
# Compile sd.cpp sources directly against our shared ggml
add_library(stable-diffusion STATIC
  engine/diffusion/stable-diffusion.cpp
  engine/diffusion/model.cpp
  engine/diffusion/name_conversion.cpp
  engine/diffusion/tokenize_util.cpp
  engine/diffusion/upscaler.cpp
  engine/diffusion/util.cpp
  engine/diffusion/version.cpp
  engine/diffusion/zip.c
  engine/diffusion/vocab/vocab.cpp
)
target_include_directories(stable-diffusion PUBLIC
  ${CMAKE_SOURCE_DIR}/engine/diffusion
)
target_include_directories(stable-diffusion PRIVATE
  ${CMAKE_SOURCE_DIR}/engine/ggml/include
  ${CMAKE_SOURCE_DIR}/engine/ggml/src
)
target_link_libraries(stable-diffusion PRIVATE ggml)
target_compile_features(stable-diffusion PRIVATE cxx_std_17)
set_target_properties(stable-diffusion PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(GPU_BACKEND STREQUAL "metal")
  target_compile_definitions(stable-diffusion PRIVATE SD_USE_METAL)
elseif(GPU_BACKEND STREQUAL "cuda")
  target_compile_definitions(stable-diffusion PRIVATE SD_USE_CUDA)
endif()
if(MSVC)
  target_compile_definitions(stable-diffusion PRIVATE
    _CRT_SECURE_NO_WARNINGS
    _USE_MATH_DEFINES
    _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
  )
  target_compile_definitions(qwen3_tts PRIVATE _CRT_SECURE_NO_WARNINGS _USE_MATH_DEFINES)
  target_compile_definitions(whisper PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# --- Unified omni.node addon ---
set(OMNI_SOURCES
  cpp/omni_binding.cpp
  cpp/llm_context.cpp
  cpp/stt_context.cpp
  cpp/tts_context.cpp
  cpp/image_context.cpp
)

add_library(omni SHARED ${OMNI_SOURCES} ${CMAKE_JS_SRC})
set_target_properties(omni PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  # MSVC multi-config generators append a per-config subdirectory (e.g. Release/)
  # to OUTPUT_DIRECTORY, causing build/Release/Release/. Setting the per-config
  # property prevents this.
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release"
)

target_include_directories(omni PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${CMAKE_SOURCE_DIR}/cpp/common
  ${CMAKE_SOURCE_DIR}/engine/include
  ${CMAKE_SOURCE_DIR}/engine/common
  ${CMAKE_SOURCE_DIR}/engine/stt
  ${CMAKE_SOURCE_DIR}/engine/tts
  ${CMAKE_SOURCE_DIR}/engine/diffusion
  ${CMAKE_SOURCE_DIR}/engine/vendor/stb
)

target_link_libraries(omni PRIVATE
  ${CMAKE_JS_LIB}
  llama
  common
  whisper
  qwen3_tts
  stable-diffusion
)

target_compile_definitions(omni PRIVATE
  NAPI_VERSION=9
  NODE_ADDON_API_DISABLE_DEPRECATED
)

# On Windows, PRIVATE link deps of static libraries (like CUDA::cuda_driver in
# ggml-cuda) don't propagate to the final shared library. Link them explicitly.
if(GPU_BACKEND STREQUAL "cuda")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(omni PRIVATE CUDA::cudart CUDA::cublas CUDA::cuda_driver)
endif()
