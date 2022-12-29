set(TMPLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TMPLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include_directories(BEFORE
  "${TMPLANG_SOURCE_DIR}/include"
  "${TMPLANG_BINARY_DIR}/include"
)

# Configure MLIR
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --src-root
set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include ) # --includedir
set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/tools/mlir/include)
include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

# TMPLANG VERSION
set(TMPLANG_VERSION_MAJOR 0)
set(TMPLANG_VERSION_MINOR 1)
set(TMPLANG_VERSION_PATCH 0)
set(TMPLANG_VERSION_STR "${TMPLANG_VERSION_MAJOR}.${TMPLANG_VERSION_MINOR}.${TMPLANG_VERSION_PATCH}")
message(STATUS "tmplang version: ${TMPLANG_VERSION}")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)
