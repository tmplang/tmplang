#ifndef TMPLANG_TREE_SOURCE_COMPILATIONUNIT_H
#define TMPLANG_TREE_SOURCE_COMPILATIONUNIT_H

#include <tmplang/Tree/Source/Decls.h>

#include <memory>

namespace tmplang::source {

class Type;

/// Represents the result of a successfully parsed source file and it is the
/// root node of every SourceTree. This class contains the ownership of every
/// declaration found in the source file.

class CompilationUnit : Node {
public:
  CompilationUnit() : Node(Node::Kind::CompilationUnit) {}

  void addFunctionDecl(FunctionDecl funcDecl) {
    FunctionDeclarations.push_back(std::move(funcDecl));
  }

  SourceLocation getBeginLoc() const override;
  SourceLocation getEndLoc() const override;

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::CompilationUnit;
  }

private:
  std::vector<FunctionDecl> FunctionDeclarations;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_COMPILATIONUNIT_H
