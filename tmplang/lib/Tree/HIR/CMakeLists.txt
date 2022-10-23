llvm_add_library(TmplangHIRTree
    HIRBuilder.cpp
    HIRContext.cpp
    Types.cpp

    LINK_COMPONENTS
    Support
)
