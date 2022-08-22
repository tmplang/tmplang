llvm_add_library(TmplangParser
    Parser.cpp

    LINK_LIBS PUBLIC
    TmplangLexer

    LINK_COMPONENTS
    Support
)
