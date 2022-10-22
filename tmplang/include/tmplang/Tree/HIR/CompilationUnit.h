#ifndef TMPLANG_TREE_HIR_COMPILATIONUNIT_H
#define TMPLANG_TREE_HIR_COMPILATIONUNIT_H

#include <tmplang/Tree/HIR/Decls.h>

namespace tmplang::hir {

/// Represents the result of a successfully compiled source file and it is the
/// root node of every AST. This class contains the ownership of every
/// declaration found in the source file.
class CompilationUnit : public Node {
public:
  CompilationUnit() : Node(Node::Kind::CompilationUnit) {}

  void addFunctionDecl(FunctionDecl funcDecl) {
    FunctionDecls.push_back(std::move(funcDecl));
  }

private:
  std::vector<FunctionDecl> FunctionDecls;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_COMPILATIONUNIT_H
