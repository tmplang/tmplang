#ifndef TMPLANG_TREE_SOURCE_DECL_H
#define TMPLANG_TREE_SOURCE_DECL_H

#include <llvm/ADT/SmallString.h>
#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/Node.h>

namespace tmplang::source {

class Decl : public Node {
public:
  virtual ~Decl() = default;

  /// All declarations have a name
  virtual llvm::StringRef getName() const = 0;

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::FuncDecl ||
           node->getKind() == Node::Kind::ParamDecl;
  }

protected:
  explicit Decl(Node::Kind k) : Node(k) {}
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_DECL_H
