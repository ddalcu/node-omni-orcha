cmake_minimum_required(VERSION 3.15)

# Find node-addon-api headers
execute_process(
  COMMAND node -p "require('node-addon-api').include"
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE NODE_ADDON_API_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

set(IMAGE_SOURCES
  image_binding.cpp
)

add_library(image SHARED ${IMAGE_SOURCES} ${CMAKE_JS_SRC})
set_target_properties(image PROPERTIES
  PREFIX ""
  SUFFIX ".node"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Release"
)

target_include_directories(image PRIVATE
  ${CMAKE_JS_INC}
  ${NODE_ADDON_API_DIR}
  ${CMAKE_SOURCE_DIR}/cpp/common
  ${CMAKE_SOURCE_DIR}/deps/stable-diffusion.cpp/include
  ${CMAKE_SOURCE_DIR}/deps/stable-diffusion.cpp/thirdparty
)

target_link_libraries(image PRIVATE
  ${CMAKE_JS_LIB}
  stable-diffusion
)

target_compile_definitions(image PRIVATE
  NAPI_VERSION=9
  NODE_ADDON_API_DISABLE_DEPRECATED
)
