#ifndef TMPLANG_OPS
#define TMPLANG_OPS

include "tmplang/Lowering/Dialect/IR/TmplangBase.td"
include "tmplang/Lowering/Dialect/IR/TmplangTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for tmplang dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Tmplang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tmplang_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def SubprogramOp : Tmplang_Op<"subprogram",
                              [FunctionOpInterface, IsolatedFromAbove, Symbol]> {
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

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    StrAttr:$sym_visibility
  );

  let regions = (region AnyRegion:$body);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::FunctionType":$type,
                   "mlir::SymbolTable::Visibility":$visibility)>
  ];
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// This function is needed so we can use FunctionOpInterface. It expects
    /// getFunctionType() to return mlir::FunctionType
    mlir::FunctionType getFunctionType();

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes();

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes();
  }];

  let hasCustomAssemblyFormat = 1;
}

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
def ReturnOp : Tmplang_Op<"return", [NoMemoryEffect, HasParent<"SubprogramOp">,
                                     Terminator]> {
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

  // Verify that the return operation must match the return type of the
  // enclosing function.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AggregateDataAccess
//===----------------------------------------------------------------------===//
def AggregateDataAccessOp : Tmplang_Op<"aggregateDataAccess", [NoMemoryEffect]> {
  let summary = "aggregate data access operation";

  let description = [{
    The "aggregateDataAccess" operation represents an access to a field of a DataType.

    ```mlir
       tmplang.aggregateDataAccess(%0), 0 : (!tmplang<data "A"{i32, i32}>) -> i32
    ```
  }];

  let arguments = (ins AggregateDataType:$input, IndexAttr:$idx);
  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;
}

#endif // TMPLANG_OPS
