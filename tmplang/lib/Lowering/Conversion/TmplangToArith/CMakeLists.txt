add_mlir_conversion_library(TmplangToArith
  TmplangToArith.cpp

  ADDITIONAL_HEADER_DIRS
  ${TMPLANG_CONVERSION_DIR}/TmplangToArith

  DEPENDS
  TmplangConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  TmplangMLIRDialect
  MLIRArithDialect
)

