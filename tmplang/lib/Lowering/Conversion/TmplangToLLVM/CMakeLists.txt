add_mlir_conversion_library(TmplangToLLVM
  TmplangToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${TMPLANG_CONVERSION_DIR}/TmplangToLLVM

  DEPENDS
  TmplangConversionPassIncGen

  LINK_LIBS PRIVATE 

  # MLIR Conversions
  MLIRFuncToLLVM

  TmplangMLIRDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRArithDialect
  MLIRTransforms

  LINK_COMPONENTS
  Core
)
