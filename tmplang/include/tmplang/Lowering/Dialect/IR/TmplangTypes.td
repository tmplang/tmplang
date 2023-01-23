#ifndef TMPLANG_TYPES
#define TMPLANG_TYPES

include "tmplang/Lowering/Dialect/IR/TmplangBase.td"

include "mlir/IR/AttrTypeBase.td"

class Tmplang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tmplang_Dialect, name, traits, "tmplang::TmplangType"> {
  let mnemonic = typeMnemonic;
}

def DataType : Tmplang_Type<"Data", "data", []> {
  let summary = "User-defined data.";

  let description = [{
    User-defined data that is is composed of a collection of other types.
  }];

  let parameters = (ins StringRefParameter<>:$name,
                        ArrayRefParameter<"mlir::Type">:$tys);

  let assemblyFormat = "$name`{` $tys `}`";
}

#endif // TMPLANG_TYPES
