add_mlir_interface(TmplangSymbol)

set(LLVM_TARGET_DEFINITIONS TmplangHIROps.td)
mlir_tablegen(TmplangHIROpsEnum.h.inc -gen-enum-decls)
mlir_tablegen(TmplangHIROpsEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRTmplangHIROpsEnumIncGen)

set(LLVM_TARGET_DEFINITIONS TmplangHIROps.td)
mlir_tablegen(TmplangHIROpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=tmplang)
mlir_tablegen(TmplangHIROpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tmplang)
add_public_tablegen_target(MLIRTmplangHIROpsAttributesIncGen)

add_mlir_dialect(TmplangHIROps tmplang)
