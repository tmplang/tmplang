llvm_add_library(TmplangHIRTree
    Decls.cpp
    HIRBuilder.cpp
    HIRContext.cpp
    Types.cpp

    LINK_COMPONENTS
    Support
)
