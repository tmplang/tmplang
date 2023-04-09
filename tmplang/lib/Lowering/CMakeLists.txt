add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Transforms)

llvm_add_library(TmplangLowering
    Lowering.cpp

    LINK_LIBS PUBLIC
    # Tmplang libraries
    TmplangHIRTree
    TmplangMLIRDialect

    # Tmplang Conversions
    TmplangToArith
    TmplangToFunc
    TmplangToLLVM

    # Tmplang Transformations
    TmplangTransforms

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
