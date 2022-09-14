llvm_add_library(TmplangAST
    ASTContext.cpp
    CompilationUnit.cpp
    Types.cpp

    LINK_COMPONENTS
    Support
)
