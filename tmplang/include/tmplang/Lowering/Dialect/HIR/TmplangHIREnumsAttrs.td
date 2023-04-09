#ifndef TMPLANG_ENUMS
#define TMPLANG_ENUMS

include "tmplang/Lowering/Dialect/HIR/TmplangHIRBase.td"

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// SymbolKinds
//===----------------------------------------------------------------------===//
def SymbolKind_Unresolved : I32BitEnumAttrCaseNone<"Unresolved">;
def SymbolKind_Subprogram : I32BitEnumAttrCaseBit<"Subprogram", 1>;
def SymbolKind_VarDecl : I32BitEnumAttrCaseBit<"VarDecl", 2>;
def SymbolKind_BuiltinType : I32BitEnumAttrCaseBit<"BuiltinType", 3>;
def SymbolKind_DataType : I32BitEnumAttrCaseBit<"DataType", 4>;
def SymbolKind_DataField : I32BitEnumAttrCaseBit<"DataField", 5>;
def SymbolKind_Type : I32BitEnumAttrCaseGroup<"Type",
  [SymbolKind_BuiltinType, SymbolKind_DataType]
>;

def SymbolKind : I32BitEnumAttr<
    "SymbolKind",
    "Tmplang symbol's kind", [
      SymbolKind_Unresolved,
      SymbolKind_Subprogram,
      SymbolKind_VarDecl,
      SymbolKind_BuiltinType,
      SymbolKind_DataType,
      SymbolKind_DataField,
      SymbolKind_Type
    ]> {

  let cppNamespace = "::tmplang";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

// All of the attributes will extend this class.
class Tmplang_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<TmplangHIRDialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def SymbolKindAttr : EnumAttr<TmplangHIRDialect, SymbolKind, "sym_kind">;

def SymbolAttr : Tmplang_Attr<"Symbol", "symbol"> {
  let parameters = (ins
    "mlir::SymbolRefAttr":$name,
    "SymbolKindAttr":$kind,
    "mlir::TypeAttr":$type
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // TMPLANG_ENUMS
