llvm_add_library(TmplangSema
    Sema.cpp
    SymbolRedefinitionValidation.cpp
    SymbolTypeResolver.cpp
    SymbolUsersAndDerivedResolver.cpp

    LINK_LIBS PUBLIC
    TmplangSupport
    TmplangMLIRDialect

    LINK_COMPONENTS
    Support
)
