set(TMPLANG_TRANSFORMS_DIR
    ${LLVM_MAIN_SRC_DIR}/../tmplang/include/tmplang/Lowering/Transforms)

add_mlir_library(TmplangTransforms
  RemoveOpsAfterReturn.cpp

  ADDITIONAL_HEADER_DIRS
  ${TMPLANG_TRANSFORMS_DIR}

  DEPENDS
  TmplangOptTransformsPassIncGen

  LINK_LIBS PUBLIC
  TmplangMLIRDialect
  MLIRAnalysis
  MLIRPass
  MLIRSupport
)
