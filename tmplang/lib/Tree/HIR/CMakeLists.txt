llvm_add_library(TmplangHIRTree
    Decls.cpp
    HIRBuilder.cpp
    HIRContext.cpp
    HIRNodeTextPrinter.cpp
    Node.cpp
    Types.cpp

    LINK_LIBS PUBLIC
    TmplangSupport

    LINK_COMPONENTS
    Support
)
