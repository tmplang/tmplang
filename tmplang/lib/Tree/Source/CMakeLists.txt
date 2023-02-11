llvm_add_library(TmplangSourceTree
    Exprs.cpp
    SourceNodeTextPrinter.cpp

    LINK_LIBS PUBLIC
    TmplangSupport
    TmplangLexer

    LINK_COMPONENTS
    Support
)
