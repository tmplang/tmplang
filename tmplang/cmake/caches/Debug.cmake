include("${CMAKE_CURRENT_LIST_DIR}/Common.cmake")

# Enable Debug build
set(CMAKE_BUILD_TYPE    "Debug" CACHE STRING "" FORCE)

# Enable sanitizers
set(LLVM_USE_SANITIZER  "Undefined;Address" CACHE STRING "" FORCE)

# Use shared libraries
set (BUILD_SHARED_LIBS  ON      CACHE BOOL "" FORCE)
