llvm_add_library(TmplangParser
    Parser.cpp

    LINK_LIBS PUBLIC
    TmplangLexer
    TmplangSourceTree

    LINK_COMPONENTS
    Support
)
