llvm_add_library(TmplangAST
    ASTContext.cpp
    ASTTextPrinter.cpp
    CompilationUnit.cpp
    Types.cpp

    LINK_COMPONENTS
    Support
)
