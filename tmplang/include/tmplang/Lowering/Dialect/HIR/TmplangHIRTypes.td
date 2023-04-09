#ifndef TMPLANG_TYPES
#define TMPLANG_TYPES

include "tmplang/Lowering/Dialect/HIR/TmplangHIRBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class TmplangTypeBase<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TmplangHIRDialect, name, traits, "tmplang::TmplangType"> {
  let mnemonic = typeMnemonic;
}

def UnresolvedType : TmplangTypeBase<"Unresolved", "unresolved", []> {
  let summary = "TODO.";

  let description = [{
    TODO.
  }];

  let parameters = (ins StringRefParameter<>:$name);

  let assemblyFormat = "$name";
}

def UnresolvedSymbolType : TmplangTypeBase<"UnresolvedSymbol", "unresolvedSymbol", []> {
  let summary = "TODO.";

  let description = [{
    TODO.
  }];

  let parameters = (ins);

  let assemblyFormat = "`?`";
}

def DataType : TmplangTypeBase<"Data", "data", []> {
  let summary = "User-defined data.";

  let description = [{
    User-defined data that is is composed of a collection of other types.
  }];

  let parameters = (ins StringRefParameter<>:$name);

  let assemblyFormat = [{ $name }];
}

def AggregateDataType : AnyTypeOf<[DataType, Builtin_Tuple]>;

// TODO: Define general TmplangType
//   - TupleType
//   - DataType
//   - FunctionType
//   - I32

#endif // TMPLANG_TYPES
