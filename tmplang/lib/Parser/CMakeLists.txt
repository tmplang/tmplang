llvm_add_library(TmplangParser
    Parser.cpp

    LINK_LIBS PUBLIC
    TmplangAST
    TmplangLexer

    LINK_COMPONENTS
    Support
)
