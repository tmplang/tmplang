#ifndef TMPLANG_OPS
#define TMPLANG_OPS

include "tmplang/Lowering/Dialect/HIR/TmplangHIREnumsAttrs.td"
include "tmplang/Lowering/Dialect/HIR/TmplangHIRBase.td"
include "tmplang/Lowering/Dialect/HIR/TmplangHIRTypes.td"
include "tmplang/Lowering/Dialect/HIR/TmplangSymbol.td"
include "mlir/IR/FunctionInterfaces.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"


// Base class for tmplang dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Tmplang_Op<string mnemonic, list<Trait> traits = []> :
    Op<TmplangHIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// TranslationUnitOp
//===----------------------------------------------------------------------===//
def TranslationUnitOp : Tmplang_Op<"translationUnit", [TmplangSymbolTable]> {
  let summary = "TODO";

  let description = [{
    TODO

    Example:

    ```mlir
    tmplang.translationUnit <fileName> {

    }
    ```
  }];

  let arguments = (ins StrAttr:$fileName);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
     $fileName attr-dict `:` $body
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// BuiltinTypeOp
//===----------------------------------------------------------------------===//
def BuiltinTypeOp : Tmplang_Op<"builtinTypeOp", [TmplangSymbol]> {
  let summary = "TODO";

  let description = [{
    TODO
  }];

  let arguments = (ins SymbolAttr:$symbol);

  let assemblyFormat = [{
    $symbol attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DataDeclOp
//===----------------------------------------------------------------------===//
def DataDeclOp : Tmplang_Op<"dataDeclOp", [TmplangSymbolTable, TmplangSymbol]> {
  let summary = "TODO";

  let description = [{
    TODO
  }];

  let arguments = (ins SymbolAttr:$symbol);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $symbol $body attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// DataFieldDeclOp
//===----------------------------------------------------------------------===//
def DataFieldDeclOp : Tmplang_Op<"dataFieldDeclOp", [TmplangSymbol]> {
  let summary = "TODO";

  let description = [{
    TODO
  }];

  let arguments = (ins SymbolAttr:$symbol);

  let assemblyFormat = [{
    $symbol attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SubprogramOp
//===----------------------------------------------------------------------===//
def SubprogramOp : Tmplang_Op<"subprogram", [TmplangSymbolTable, TmplangSymbol]> {
  let summary = "user defined function operation";

  let description = [{
    The "tmplang.func" operation represents a user defined function. These are
    callable SSA-region operations that contain tmplang computations.

    Example:

    ```mlir
    tmplang.func @main() {

    }
    ```
  }];

  let arguments = (ins SymbolAttr:$symbol);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//
    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes();

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes();
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SubprogramParamOp
//===----------------------------------------------------------------------===//
def SubprogramParamOp : Tmplang_Op<"subprogramParam", [TmplangSymbol]> {
  let summary = "Represents a parameter of a subprogram";

  let description = [{
    This operation represetns a parameter of a subprogram. It helps in the
    creation of an alloca for each parameter. For example:

    ```mlir
      %0 = tmplang.subprogramParam(%param) : i32
    ```
  }];

  let arguments = (ins SymbolAttr:$symbol);
  let results = (outs);

  let assemblyFormat = [{
    $symbol attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
def ConstantOp : Tmplang_Op<"constant", [NoMemoryEffect]> {
  let summary = "constant operation";

  let description = [{
    The "constant" operation builds a SSA value given a constant value. For example:

    ```mlir
      tmplang.constant 0 : i32
    ```
  }];

  let arguments = (ins AnyAttrOf<[I32Attr]>:$value);
  let results = (outs AnyTypeOf<[I32]>:$result);

  let builders = [
    OpBuilder<(ins "mlir::IntegerAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
  ];

  bit hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// TupleOp 
//===----------------------------------------------------------------------===//
def TupleOp : Tmplang_Op<"tuple", [NoMemoryEffect]> {
  let summary = "tuple operation";

  let description = [{
    The "tuple" operation builds a SSA value given a compedium of several
    constant values. For example:

    ```mlir
      tmplang.tuple () : tuple<>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs Builtin_Tuple:$result);

  let assemblyFormat = [{
    `(` $values `)` attr-dict `:` functional-type($values, results)
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
def ReturnOp : Tmplang_Op<"return", [NoMemoryEffect]> {
  let summary = "return operation";

  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.

    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      tmplang.func @foo() -> i32 {
        ...
        tmplang.return(%0) -> i32
      }
    ```
  }];

  let arguments = (ins Optional<AnyType>:$input);
  let results = (outs); 

  let builders = [
    // Allow building a ReturnOp with no return operand.
    OpBuilder<(ins), [{ build($_builder, $_state, mlir::Value{}); }]>,
  ];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let assemblyFormat = "attr-dict (`(`$input^`)` `->` type($input))?";
}

//===----------------------------------------------------------------------===//
// VarRefOp
//===----------------------------------------------------------------------===//
def VarRefOp : Tmplang_Op<"varRef", [TmplangSymbolUser]> {
  let summary = "TODO";

  let description = [{
    TODO
  }];

  let arguments = (ins SymbolRefAttr:$symbol_ref);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $symbol_ref attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// TupleAccess
//===----------------------------------------------------------------------===//
def TupleAccessOp : Tmplang_Op<"tupleAccess", [NoMemoryEffect]> {
  let summary = "aggregate data access operation";

  let description = [{
    The "aggregateDataAccess" operation represents an access to a field of a DataType.

    ```mlir
       tmplang.aggregateDataAccess(%0), 0 : (!tmplang<data "A"{i32, i32}>) -> i32
    ```
  }];

  let arguments = (ins AggregateDataType:$input, IndexAttr:$idx);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $input `)` $idx attr-dict `:` functional-type($input, results)
  }];
}

//===----------------------------------------------------------------------===//
// DataAccess
//===----------------------------------------------------------------------===//
def DataAccessOp : Tmplang_Op<"dataAccess", [NoMemoryEffect, TmplangSymbolUser]> {
  let summary = "aggregate data access operation";

  let description = [{
    The "aggregateDataAccess" operation represents an access to a field of a DataType.

    ```mlir
       tmplang.aggregateDataAccess(%0), 0 : (!tmplang<data "A"{i32, i32}>) -> i32
    ```
  }];

  let arguments = (ins AggregateDataType:$input, SymbolRefAttr:$symbol_ref);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $input `)` $symbol_ref attr-dict `:` functional-type($input, results)
  }];
}

//===----------------------------------------------------------------------===//
// MatchExpr
//===----------------------------------------------------------------------===//
def MatchOp : Tmplang_Op<"match", [NoMemoryEffect]> {
  let summary = "match operation";

  let description = [{
    The "match" operation represents patter-mattching access to a value.

    ```mlir
       %0 = "tmplang.match"(%arg0) ({
        
       }) : (tuple<i32, !tmplang<data "Dummy"{i32, i32}>>) -> i32
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// MatchYieldOp
//===----------------------------------------------------------------------===//
def MatchYieldOp : Tmplang_Op<"match_yield", [NoMemoryEffect, ReturnLike,
                                              Terminator, HasParent<"MatchOp">]> {
  let summary = "match yield and termination operation";

  let description = [{
    TODO: Add our documentation

    "match_yield" yields an SSA value from the match op region and terminates
    the regions. If "match_yield" has any operands, the operands must match the
    parent operation's results.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict (`(`$results^`)` `:` type($results))? }];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Verify that the return operation must match the return type of the
  // enclosing match.
  let hasVerifier = 1;
}

#endif // TMPLANG_OPS
