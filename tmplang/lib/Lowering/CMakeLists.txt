add_subdirectory(Dialect)
add_subdirectory(Conversion)

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

    # MLIR Conversions
    MLIRLLVMCommonConversion
    MLIRArithToLLVM
    MLIRLLVMToLLVMIRTranslation

    # MLIR Transformations

    # MLIR Support Libraries
    MLIRIR
    MLIRPass
    MLIRSupport

    LINK_COMPONENTS
    Support
)
