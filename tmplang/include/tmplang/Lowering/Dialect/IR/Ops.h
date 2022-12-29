#ifndef TMPLANG_LOWERING_DIALECT_IR_OPS_H
#define TMPLANG_LOWERING_DIALECT_IR_OPS_H

#include <tmplang/Lowering/Dialect/IR/Dialect.h>
#include <tmplang/Lowering/Dialect/IR/Types.h>

/// Include the auto-generated header file containing the declarations of the
/// tmplang operations.
#define GET_OP_CLASSES
#include <tmplang/Lowering/Dialect/IR/TmplangOps.h.inc>

#endif // TMPLANG_LOWERING_DIALECT_IR_OPS_H
