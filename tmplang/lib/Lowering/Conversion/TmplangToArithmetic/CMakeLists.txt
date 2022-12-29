add_mlir_conversion_library(TmplangToArithmetic
  TmplangToArithmetic.cpp

  ADDITIONAL_HEADER_DIRS
  ${TMPLANG_CONVERSION_DIR}/TmplangToArithmetic

  DEPENDS
  TmplangConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  TmplangMLIRDialect
  MLIRArithmeticDialect
)

