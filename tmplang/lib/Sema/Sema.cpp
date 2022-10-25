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

  bool visitFunctionDecl(const FunctionDecl &funcDecl) {
    auto *builtinType = llvm::dyn_cast<BuiltinType>(&funcDecl.getReturnType());
    if (!builtinType || builtinType->getBuiltinKind() != BuiltinType::K_Unit) {
      return false;
    }

    return true;
  }
};

} // namespace

bool tmplang::Sema(CompilationUnit &compUnit) {
  return SemaAnalyzerVisitor{}.traverseNode(compUnit);
}
