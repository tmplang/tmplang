llvm_add_library(TmplangSema
    Sema.cpp

    LINK_LIBS PUBLIC
    TmplangDiagnostics
    TmplangHIRTree

    LINK_COMPONENTS
    Support
)
