#ifndef TMPLANG_TREE_HIR_HIRBUILDER_H
#define TMPLANG_TREE_HIR_HIRBUILDER_H

#include <tmplang/Tree/HIR/CompilationUnit.h>
#include <tmplang/Tree/HIR/HIRContext.h>

// Forward declarations
namespace tmplang::source {
class CompilationUnit;
} // namespace tmplang::source

namespace tmplang::hir {

/// Builds a CompilationUnit contaning all semantic information. This tree is
/// not yet verified to be semantic correct.
std::optional<CompilationUnit> buildHIR(const source::CompilationUnit &,
                                   HIRContext &);

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_HIRBUILDER_H
