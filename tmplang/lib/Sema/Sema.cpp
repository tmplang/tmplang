#include <tmplang/Sema/Sema.h>

#include <tmplang/Tree/HIR/CompilationUnit.h>
#include <tmplang/Tree/HIR/RecursiveNodeVisitor.h>
#include <tmplang/Tree/HIR/Types.h>

using namespace tmplang;
using namespace tmplang::hir;

namespace {

class SemaAnalyzerVisitor : public RecursiveASTVisitor<SemaAnalyzerVisitor> {
public:
  using Base = RecursiveASTVisitor<SemaAnalyzerVisitor>;
};

} // namespace

bool tmplang::Sema(CompilationUnit &compUnit) {
  // At the moment, it does nothing
  return SemaAnalyzerVisitor{}.traverseNode(compUnit);
}
