llvm_add_library(TmplangLexer
    Lexer.cpp
    Token.cpp

    LINK_LIBS PUBLIC
    TmplangSupport

    LINK_COMPONENTS
    Support
)
