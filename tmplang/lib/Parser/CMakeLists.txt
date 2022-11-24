llvm_add_library(TmplangParser
    Parser.cpp

    LINK_LIBS PUBLIC
    TmplangLexer
    TmplangSourceTree
    TmplangDiagnostics

    LINK_COMPONENTS
    Support
)
