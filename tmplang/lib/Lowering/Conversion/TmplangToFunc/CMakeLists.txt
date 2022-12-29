add_mlir_conversion_library(TmplangToFunc
  TmplangToFunc.cpp

  ADDITIONAL_HEADER_DIRS
  ${TMPLANG_CONVERSION_DIR}/TmplangToFunc

  DEPENDS
  TmplangConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  TmplangMLIRDialect
  MLIRFuncDialect
)

