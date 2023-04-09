#ifndef TMPLANG_LOWERING_DIALECT_HIR_DIALECT_H
#define TMPLANG_LOWERING_DIALECT_HIR_DIALECT_H

#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

/// Include the auto-generated header file containing the declaration of the
/// tmplang dialect.
#include <tmplang/Lowering/Dialect/HIR/TmplangHIROpsDialect.h.inc>

#endif // TMPLANG_LOWERING_DIALECT_HIR_DIALECT_H
