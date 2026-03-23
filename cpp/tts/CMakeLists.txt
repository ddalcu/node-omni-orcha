cmake_minimum_required(VERSION 3.15)

execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

# TTS_BUILD_DIR is set by the parent CMakeLists.txt ExternalProject

set(TTS_SOURCES
  tts_binding.cpp
)

add_library(tts_addon SHARED ${TTS_SOURCES} ${CMAKE_JS_SRC})
set_target_properties(tts_addon PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  OUTPUT_NAME "tts"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)

# Must wait for external project to finish building
add_dependencies(tts_addon tts_external)

target_include_directories(tts_addon PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${CMAKE_SOURCE_DIR}/cpp/common
  ${CMAKE_SOURCE_DIR}/deps/TTS.cpp/include
  ${CMAKE_SOURCE_DIR}/deps/TTS.cpp/src/models
)

target_link_libraries(tts_addon PRIVATE
  ${CMAKE_JS_LIB}
  "${TTS_BUILD_DIR}/src/libtts.a"
  "${TTS_BUILD_DIR}/ggml/src/libggml.a"
  "${TTS_BUILD_DIR}/ggml/src/ggml-cpu/libggml-cpu.a"
  "${TTS_BUILD_DIR}/ggml/src/ggml-blas/libggml-blas.a"
  "${TTS_BUILD_DIR}/ggml/src/libggml-base.a"
)

if(APPLE)
  target_link_libraries(tts_addon PRIVATE
    "-framework Accelerate"
  )
endif()

target_compile_definitions(tts_addon PRIVATE
  NAPI_VERSION=9
  NODE_ADDON_API_DISABLE_DEPRECATED
)
