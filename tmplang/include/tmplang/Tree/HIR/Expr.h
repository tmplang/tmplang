#ifndef TMPLANG_TREE_HIR_EXPR_H
#define TMPLANG_TREE_HIR_EXPR_H

#include <tmplang/Tree/HIR/Node.h>

namespace tmplang::hir {

class Type;

class Expr : public Node {
public:
  virtual ~Expr() = default;

  virtual const Type &getType() const { return Ty; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprIntegerNumber;
  }

protected:
  explicit Expr(Node::Kind k, const source::Node &srcNode, const Type &ty)
      : Node(k, srcNode), Ty(ty) {}

private:
  const Type &Ty;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_EXPR_H
