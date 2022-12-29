#include <tmplang/Lowering/Dialect/IR/Dialect.h>

#include <tmplang/Lowering/Dialect/IR/Ops.h>
#include <tmplang/Lowering/Dialect/IR/Types.h>

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void tmplang::TmplangDialect::initialize() {
  registerOps();
  registerTypes();
}

/// Include all auto generated code from the dialect
#include <tmplang/Lowering/Dialect/IR/TmplangOpsDialect.cpp.inc>
