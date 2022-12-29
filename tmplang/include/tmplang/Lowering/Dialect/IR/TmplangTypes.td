#ifndef TMPLANG_TYPES
#define TMPLANG_TYPES

include "tmplang/Lowering/Dialect/IR/TmplangBase.td"

include "mlir/IR/AttrTypeBase.td"

class Tmplang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tmplang_Dialect, name, traits, "tmplang::TmplangType"> {
  let mnemonic = typeMnemonic;
}

#endif // TMPLANG_TYPES
