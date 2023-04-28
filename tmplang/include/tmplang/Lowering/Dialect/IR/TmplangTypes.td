#ifndef TMPLANG_TYPES
#define TMPLANG_TYPES

include "tmplang/Lowering/Dialect/IR/TmplangBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class TmplangTypeBase<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tmplang_Dialect, name, traits, "tmplang::TmplangType"> {
  let mnemonic = typeMnemonic;
}

def DataType : TmplangTypeBase<"Data", "data", []> {
  let summary = "User-defined data.";

  let description = [{
    User-defined data that is is composed of a collection of other types.
  }];

  let parameters = (ins StringRefParameter<>:$name,
                        ArrayRefParameter<"mlir::Type">:$tys);

  let assemblyFormat = "$name`{` $tys `}`";
}

def UnionType : TmplangTypeBase<"Union", "Union", []> {
  let summary = "Discriminated union of different alternatives.";

  let description = [{
    Discriminated union of different alternatives of data.
  }];

  let parameters = (ins StringRefParameter<>:$name,
                        ArrayRefParameter<"tmplang::DataType">:$tys);

  let assemblyFormat = "$name`{` $tys `}`";
}

def AggregateDataType : AnyTypeOf<[DataType, Builtin_Tuple]>;

// TODO: Define general TmplangType
//   - TupleType
//   - DataType
//   - FunctionType
//   - I32

#endif // TMPLANG_TYPES
