llvm_add_library(TmplangHIRTree
    Exprs.cpp
    Decls.cpp
    HIRBuilder.cpp
    HIRContext.cpp
    HIRNodeTextPrinter.cpp
    Node.cpp
    Symbol.cpp
    Types.cpp

    LINK_LIBS PUBLIC
    TmplangLexer
    TmplangSourceTree
    TmplangSupport

    LINK_COMPONENTS
    Support
)
