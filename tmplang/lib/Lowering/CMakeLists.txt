add_subdirectory(Dialect)
add_subdirectory(Sema)

llvm_add_library(TmplangLowering
    HIRBuilder.cpp

    LINK_LIBS PUBLIC
    # Tmplang libraries
    TmplangSourceTree
    TmplangSema
    TmplangMLIRDialect

    # Tmplang Conversions

    # Tmplang Transformations

    # MLIR Conversions
    MLIRLLVMCommonConversion
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation

    # MLIR Transformations

    # MLIR Support Libraries
    MLIRIR
    MLIRPass
    MLIRSupport

    LINK_COMPONENTS
    Support
)
