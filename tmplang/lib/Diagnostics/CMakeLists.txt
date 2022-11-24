llvm_add_library(TmplangDiagnostics
    Diagnostic.cpp
    DiagnosticPrettyPrinter.cpp
    Hint.cpp

    LINK_LIBS PUBLIC
    TmplangSupport

    LINK_COMPONENTS
    Support
)
