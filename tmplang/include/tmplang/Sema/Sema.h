#ifndef TMPLANG_SEMA_SEMA_H
#define TMPLANG_SEMA_SEMA_H

namespace tmplang {

namespace hir {
class CompilationUnit;
} // namespace hir

/// Semantic analysis. Right now it just does some restrictive verifications
/// before lowering to MLIR. Does not report anything.
bool Sema(tmplang::hir::CompilationUnit &compUnit);

} // namespace tmplang

#endif // TMPLANG_SEMA_SEMA_H
