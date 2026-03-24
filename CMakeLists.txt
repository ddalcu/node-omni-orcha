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
    # Check for CUDA toolkit
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(GPU_BACKEND "cuda")
    else()
      set(GPU_BACKEND "cpu")
    endif()
  endif()
  message(STATUS "[node-omni-orcha] Auto-detected GPU backend: ${GPU_BACKEND}")
endif()

# Build static libraries from deps
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# --- llama.cpp ---
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/llama.cpp/CMakeLists.txt")
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)

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

  add_subdirectory(deps/llama.cpp)
  message(STATUS "[node-omni-orcha] llama.cpp configured with GPU_BACKEND=${GPU_BACKEND}")
endif()

# --- stable-diffusion.cpp ---
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/stable-diffusion.cpp/CMakeLists.txt")
  set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

  if(GPU_BACKEND STREQUAL "metal")
    set(SD_METAL ON CACHE BOOL "" FORCE)
    set(SD_CUDA OFF CACHE BOOL "" FORCE)
  elseif(GPU_BACKEND STREQUAL "cuda")
    set(SD_METAL OFF CACHE BOOL "" FORCE)
    set(SD_CUDA ON CACHE BOOL "" FORCE)
  else()
    set(SD_METAL OFF CACHE BOOL "" FORCE)
    set(SD_CUDA OFF CACHE BOOL "" FORCE)
  endif()

  add_subdirectory(deps/stable-diffusion.cpp)
  message(STATUS "[node-omni-orcha] stable-diffusion.cpp configured with GPU_BACKEND=${GPU_BACKEND}")
endif()

# --- whisper.cpp ---
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/whisper.cpp/CMakeLists.txt")
  set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)

  # whisper.cpp uses ggml internally with its own GPU detection
  # We set the same GGML flags it respects
  add_subdirectory(deps/whisper.cpp)
  message(STATUS "[node-omni-orcha] whisper.cpp configured with GPU_BACKEND=${GPU_BACKEND}")
endif()

# --- Engine .node addons ---
add_subdirectory(cpp/llm)

if(TARGET stable-diffusion)
  add_subdirectory(cpp/image)
endif()

# --- TTS.cpp (built as external project to avoid ggml target conflicts) ---
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/TTS.cpp/CMakeLists.txt")
  include(ExternalProject)
  set(TTS_BUILD_DIR "${CMAKE_BINARY_DIR}/tts-build")

  set(TTS_PLATFORM_BYPRODUCTS "")
  if(APPLE)
    list(APPEND TTS_PLATFORM_BYPRODUCTS
      "${TTS_BUILD_DIR}/ggml/src/ggml-metal/libggml-metal.a"
      "${TTS_BUILD_DIR}/ggml/src/ggml-blas/libggml-blas.a"
    )
  endif()
  ExternalProject_Add(tts_external
    SOURCE_DIR "${CMAKE_SOURCE_DIR}/deps/TTS.cpp"
    BINARY_DIR "${TTS_BUILD_DIR}"
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
      -DBUILD_SHARED_LIBS=OFF
      -DTTS_BUILD_EXAMPLES=OFF
      -DCMAKE_CXX_STANDARD=17
    BUILD_BYPRODUCTS
      "${TTS_BUILD_DIR}/src/libtts.a"
      "${TTS_BUILD_DIR}/ggml/src/libggml.a"
      "${TTS_BUILD_DIR}/ggml/src/libggml-base.a"
      "${TTS_BUILD_DIR}/ggml/src/ggml-cpu/libggml-cpu.a"
      ${TTS_PLATFORM_BYPRODUCTS}
    INSTALL_COMMAND ""
  )

  # Create imported targets pointing to the build output directly
  add_library(tts_lib STATIC IMPORTED)
  set_target_properties(tts_lib PROPERTIES IMPORTED_LOCATION "${TTS_BUILD_DIR}/src/libtts.a")
  add_dependencies(tts_lib tts_external)

  add_library(tts_ggml STATIC IMPORTED)
  set_target_properties(tts_ggml PROPERTIES IMPORTED_LOCATION "${TTS_BUILD_DIR}/ggml/src/libggml.a")
  add_dependencies(tts_ggml tts_external)

  add_library(tts_ggml_base STATIC IMPORTED)
  set_target_properties(tts_ggml_base PROPERTIES IMPORTED_LOCATION "${TTS_BUILD_DIR}/ggml/src/libggml-base.a")

  add_library(tts_ggml_cpu STATIC IMPORTED)
  set_target_properties(tts_ggml_cpu PROPERTIES IMPORTED_LOCATION "${TTS_BUILD_DIR}/ggml/src/ggml-cpu/libggml-cpu.a")

  if(APPLE)
    add_library(tts_ggml_metal STATIC IMPORTED)
    set_target_properties(tts_ggml_metal PROPERTIES IMPORTED_LOCATION "${TTS_BUILD_DIR}/ggml/src/ggml-metal/libggml-metal.a")

    add_library(tts_ggml_blas STATIC IMPORTED)
    set_target_properties(tts_ggml_blas PROPERTIES IMPORTED_LOCATION "${TTS_BUILD_DIR}/ggml/src/ggml-blas/libggml-blas.a")
  endif()

  set(TTS_AVAILABLE ON)
  message(STATUS "[node-omni-orcha] TTS.cpp configured as external project")
endif()

if(TARGET whisper)
  add_subdirectory(cpp/stt)
endif()

if(TTS_AVAILABLE)
  add_subdirectory(cpp/tts)
endif()

# --- qwen3-tts.cpp (pre-built externally, linked directly) ---
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/qwen3-tts.cpp/build/libqwen3_tts.a")
  add_subdirectory(cpp/tts_qwen3)
  message(STATUS "[node-omni-orcha] qwen3-tts.cpp addon enabled")
endif()
