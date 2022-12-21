#ifndef TMPLANG_TREE_HIR_EXPRS_H
#define TMPLANG_TREE_HIR_EXPRS_H

#include <llvm/ADT/APInt.h>
#include <tmplang/Tree/HIR/Expr.h>

namespace tmplang::hir {

class ExprIntegerNumber final : public Expr {
public:
  ExprIntegerNumber(const source::Node &srcNode, const Type &ty,
                    llvm::APInt num)
      : Expr(Kind::ExprIntegerNumber, srcNode, ty), Num(std::move(num)) {}

  llvm::APInt getNumber() const { return Num; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::ExprIntegerNumber;
  }

private:
  llvm::APInt Num;
};

class ExprRet final : public Expr {
public:
  ExprRet(const source::Node &srcNode, const Type &ty,
          std::unique_ptr<Expr> expr = nullptr)
      : Expr(Kind::ExprRet, srcNode, ty), ExprToRet(std::move(expr)) {}

  const Expr *getReturnedExpr() const { return ExprToRet.get(); }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprRet;
  }

private:
  std::unique_ptr<Expr> ExprToRet;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_EXPRS_H
