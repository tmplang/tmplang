llvm_add_library(TmplangSourceTree
    SourceNodeTextPrinter.cpp

    LINK_LIBS PUBLIC
    TmplangSupport
    TmplangLexer

    LINK_COMPONENTS
    Support
)
