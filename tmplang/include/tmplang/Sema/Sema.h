#ifndef TMPLANG_SEMA_SEMA_H
#define TMPLANG_SEMA_SEMA_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang {

class SourceManager;

namespace hir {
class CompilationUnit;
} // namespace hir

/// Semantic analysis. Verifies the semantic correctness of the code. If it
/// contains irregularties, they are reported though diagnostics.
bool Sema(tmplang::hir::CompilationUnit &, const SourceManager &,
          llvm::raw_ostream &);

} // namespace tmplang

#endif // TMPLANG_SEMA_SEMA_H
