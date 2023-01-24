#ifndef TMPLANG_TREE_HIR_EXPRS_H
#define TMPLANG_TREE_HIR_EXPRS_H

#include <llvm/ADT/APInt.h>
#include <tmplang/Tree/HIR/Expr.h>
#include <tmplang/Tree/HIR/Symbol.h>
#include <tmplang/Tree/HIR/Types.h>

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

class ExprTuple final : public Expr {
public:
  ExprTuple(const source::Node &srcNode, const TupleType &ty,
            SmallVector<std::unique_ptr<hir::Expr>, 4> values)
      : Expr(Kind::ExprTuple, srcNode, ty), Values(std::move(values)) {}

  ArrayRef<std::unique_ptr<Expr>> getVals() const { return Values; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprTuple;
  }

private:
  SmallVector<std::unique_ptr<Expr>, 4> Values;
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

class ExprVarRef final : public Expr {
public:
  ExprVarRef(const source::Node &srcNode, const Symbol &sym)
      : Expr(Kind::ExprVarRef, srcNode, sym.getType()), ReferencedSym(sym) {}

  llvm::StringRef getName() const { return ReferencedSym.getId(); }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprVarRef;
  }

private:
  const Symbol &ReferencedSym;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_EXPRS_H
