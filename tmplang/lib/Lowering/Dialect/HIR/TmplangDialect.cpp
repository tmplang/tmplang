#include <tmplang/Lowering/Dialect/HIR/Dialect.h>

#include <tmplang/Lowering/Dialect/HIR/Ops.h>
#include <tmplang/Lowering/Dialect/HIR/Types.h>

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void tmplang::TmplangHIRDialect::initialize() {
  registerOps();
  registerTypes();
  registerAttrs();
}

/// Include all auto generated code from the dialect
#include <tmplang/Lowering/Dialect/HIR/TmplangHIROpsDialect.cpp.inc>
