set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(TmplangOptions)

# So we can include Options.inc
include_directories(BEFORE
  "${TMPLANG_BINARY_DIR}"
)

llvm_add_library(TmplangCLI
    Arguments.cpp
    CLPrinter.cpp

    LINK_COMPONENTS
    Support
    Core
    Option
)

add_dependencies(TmplangCLI TmplangOptions)
