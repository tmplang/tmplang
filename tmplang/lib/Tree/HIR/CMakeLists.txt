llvm_add_library(TmplangHIRTree
    HIRContext.cpp
    CompilationUnit.cpp
    Types.cpp

    LINK_COMPONENTS
    Support
)
