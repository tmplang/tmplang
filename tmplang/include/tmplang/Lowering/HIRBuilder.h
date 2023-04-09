#ifndef TMPLANG_LOWERING_HIRBUILDER_H
#define TMPLANG_LOWERING_HIRBUILDER_H

#include <mlir/IR/MLIRContext.h>
#include <tmplang/Lowering/Dialect/HIR/Ops.h>

namespace tmplang {
class SourceManager;
namespace source {
class CompilationUnit;
} // namespace source

/// Lowers the source::CompilationUnit node to a TranslationUnitOp. It requires
/// the SourceManager to be able to generate debug locations.
TranslationUnitOp LowerToHIR(const source::CompilationUnit &,
                             mlir::MLIRContext &, const SourceManager &);

} // namespace tmplang

#endif // TMPLANG_LOWERING_HIRBUILDER_H
