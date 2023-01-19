#ifndef TMPLANG_TREE_SOURCE_COMPILATIONUNIT_H
#define TMPLANG_TREE_SOURCE_COMPILATIONUNIT_H

#include <tmplang/Tree/Source/Decls.h>

#include <memory>

namespace tmplang::source {

class Type;

/// Represents the result of a successfully parsed source file and it is the
/// root node of every SourceTree. This class contains the ownership of every
/// declaration found in the source file.

class CompilationUnit : public Node {
public:
  CompilationUnit(std::vector<SubprogramDecl> functions,
                  bool didRecoverFromAnError)
      : Node(Node::Kind::CompilationUnit),
        ContainsErrorRecoveryTokens(didRecoverFromAnError),
        SubprogramDeclarations(std::move(functions)) {}

  tmplang::SourceLocation getBeginLoc() const override { return InvalidLoc; }
  tmplang::SourceLocation getEndLoc() const override { return InvalidLoc; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::CompilationUnit;
  }

  ArrayRef<SubprogramDecl> getSubprogramDecls() const {
    return SubprogramDeclarations;
  }

  bool didRecoverFromAnError() const { return ContainsErrorRecoveryTokens; }

private:
  bool ContainsErrorRecoveryTokens;
  std::vector<SubprogramDecl> SubprogramDeclarations;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_COMPILATIONUNIT_H
