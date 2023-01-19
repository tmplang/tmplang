#ifndef TMPLANG_TREE_HIR_DECL_H
#define TMPLANG_TREE_HIR_DECL_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/Tree/HIR/Node.h>

namespace tmplang::source {
class Node;
} // namespace tmplang::source

namespace tmplang::hir {

class Decl : public Node {
public:
  StringRef getName() const { return Name; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::SubprogramDecl ||
           node->getKind() == Node::Kind::ParamDecl;
  }

  virtual ~Decl() = default;

protected:
  explicit Decl(Node::Kind k, const source::Node &srcNode, StringRef name)
      : Node(k, srcNode), Name(name) {}

private:
  /// All Decls have a name
  SmallString<32> Name;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_DECL_H
