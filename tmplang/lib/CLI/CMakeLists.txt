llvm_add_library(TmplangCLI
    Arguments.cpp
    CLPrinter.cpp

    LINK_COMPONENTS
    Support
    Core
    Option
)

add_dependencies(TmplangCLI TmplangOptions)
