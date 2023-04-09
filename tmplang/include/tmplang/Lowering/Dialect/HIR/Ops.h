#ifndef TMPLANG_LOWERING_DIALECT_HIR_OPS_H
#define TMPLANG_LOWERING_DIALECT_HIR_OPS_H

#include <tmplang/Lowering/Dialect/HIR/Dialect.h>
#include <tmplang/Lowering/Dialect/HIR/EnumAttrs.h>
#include <tmplang/Lowering/Dialect/HIR/Symbol.h>
#include <tmplang/Lowering/Dialect/HIR/Types.h>

/// Include the auto-generated header file containing the declarations of the
/// tmplang operations.
#define GET_OP_CLASSES
#include <tmplang/Lowering/Dialect/HIR/TmplangHIROps.h.inc>

#endif // TMPLANG_LOWERING_DIALECT_HIR_OPS_H
