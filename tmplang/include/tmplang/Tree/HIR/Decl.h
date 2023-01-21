#ifndef TMPLANG_TREE_HIR_DECL_H
#define TMPLANG_TREE_HIR_DECL_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/Tree/HIR/Node.h>

namespace tmplang::source {
class Node;
} // namespace tmplang::source

namespace tmplang::hir {

// Forward declarations
class Symbol;
class Type;

class Decl : public Node {
public:
  StringRef getName() const;
  const Symbol &getSymbol() const { return Sym; }
  virtual const Type &getType() const;

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::SubprogramDecl ||
           node->getKind() == Node::Kind::ParamDecl;
  }

  virtual ~Decl() = default;

protected:
  explicit Decl(Node::Kind k, const source::Node &srcNode, const Symbol &sym)
      : Node(k, srcNode), Sym(sym) {}

private:
  const Symbol &Sym;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_DECL_H
