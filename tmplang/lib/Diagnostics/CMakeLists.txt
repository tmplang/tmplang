llvm_add_library(TmplangDiagnostics
    Diagnostic.cpp

    LINK_LIBS PUBLIC
    TmplangSupport

    LINK_COMPONENTS
    Support
)
