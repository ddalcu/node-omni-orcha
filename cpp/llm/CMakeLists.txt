cmake_minimum_required(VERSION 3.15)

# Find node-addon-api headers
execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Strip quotes from the path
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

# Source files
set(LLM_SOURCES
  llm_binding.cpp
)

# Build llm.node
add_library(llm SHARED ${LLM_SOURCES} ${CMAKE_JS_SRC})
set_target_properties(llm PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  # Output to build/Release/ for cmake-js compatibility
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)

target_include_directories(llm PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${CMAKE_SOURCE_DIR}/cpp/common
  ${CMAKE_SOURCE_DIR}/deps/llama.cpp/include
  ${CMAKE_SOURCE_DIR}/deps/llama.cpp/common
)

target_link_libraries(llm PRIVATE
  ${CMAKE_JS_LIB}
  llama
  common
)

# N-API version
target_compile_definitions(llm PRIVATE
  NAPI_VERSION=9
  NODE_ADDON_API_DISABLE_DEPRECATED
)
