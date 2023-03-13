# Append tmplang compiler as an external project of LLVM
set(LLVM_TOOL_TMPLANG_BUILD  ON        CACHE BOOL   "" FORCE)
set(LLVM_EXTERNAL_PROJECTS   "tmplang" CACHE STRING "" FORCE)
set(LLVM_EXTERNAL_TMPLANG_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../tmplang"     CACHE STRING "" FORCE)

# Minimal required LLVM projects to build tmplang compiler
set(LLVM_ENABLE_PROJECTS     "mlir"   CACHE STRING "" FORCE)

# Use optimized tablegen
set(LLVM_OPTIMIZED_TABLEGEN  ON   CACHE BOOL "" FORCE)

# Build a few common targets
set (LLVM_TARGETS_TO_BUILD   "AArch64;ARM;WebAssembly;X86"   CACHE STRING "" FORCE)

# Always use ccache 
set(LLVM_CCACHE_BUILD        ON   CACHE BOOL "" FORCE)
