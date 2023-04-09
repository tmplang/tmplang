#ifndef TMPLANG_LOWERING_DIALECT_IR_SYMBOL
#define TMPLANG_LOWERING_DIALECT_IR_SYMBOL

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Symbol Traits
//===----------------------------------------------------------------------===//
def TmplangSymbolUser : NativeOpTrait<"SymbolUser"> {
  let cppNamespace = "::tmplang::OpTrait";
}

def TmplangSymbol : NativeOpTrait<"Symbol"> {
  let cppNamespace = "::tmplang::OpTrait";
}

// Op defines a symbol table.
def TmplangSymbolTable : NativeOpTrait<"SymbolTable"> {
  let cppNamespace = "::tmplang::OpTrait";
}

#endif // TMPLANG_LOWERING_DIALECT_IR_SYMBOL