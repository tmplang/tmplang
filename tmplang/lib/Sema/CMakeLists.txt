llvm_add_library(TmplangSema
    Sema.cpp

    LINK_LIBS PUBLIC
    TmplangHIRTree

    LINK_COMPONENTS
    Support
)
