cmake_minimum_required(VERSION 3.15)

execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

set(STT_SOURCES
  stt_binding.cpp
)

add_library(stt SHARED ${STT_SOURCES} ${CMAKE_JS_SRC})
set_target_properties(stt PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)

target_include_directories(stt PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${CMAKE_SOURCE_DIR}/cpp/common
  ${CMAKE_SOURCE_DIR}/deps/whisper.cpp/include
)

target_link_libraries(stt PRIVATE
  ${CMAKE_JS_LIB}
  whisper
)

target_compile_definitions(stt PRIVATE
  NAPI_VERSION=9
  NODE_ADDON_API_DISABLE_DEPRECATED
)
