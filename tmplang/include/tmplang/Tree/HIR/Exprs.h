#ifndef TMPLANG_TREE_HIR_EXPRS_H
#define TMPLANG_TREE_HIR_EXPRS_H

#include <llvm/ADT/APInt.h>
#include <tmplang/Tree/HIR/Expr.h>
#include <tmplang/Tree/HIR/Symbol.h>
#include <tmplang/Tree/HIR/Types.h>

#include <optional>
#include <variant>

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
  const Symbol &getSymbol() const { return ReferencedSym; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprVarRef;
  }

private:
  const Symbol &ReferencedSym;
};

class ExprAggregateDataAccess final : public Expr {
public:
  using BaseExpr =
      std::variant<std::unique_ptr<ExprAggregateDataAccess>,
                   std::unique_ptr<ExprVarRef>, std::unique_ptr<ExprTuple>>;

  static BaseExpr
  FromExprToAggregateDataAccOrVarRefOrTuple(std::unique_ptr<Expr> expr) {
    if (isa<ExprVarRef>(expr.get())) {
      return BaseExpr(std::unique_ptr<ExprVarRef>(
          static_cast<ExprVarRef *>(expr.release())));
    }
    if (isa<ExprTuple>(expr.get())) {
      return BaseExpr(
          std::unique_ptr<ExprTuple>(static_cast<ExprTuple *>(expr.release())));
    }
    assert(isa<ExprAggregateDataAccess>(expr.get()));
    return BaseExpr(std::unique_ptr<ExprAggregateDataAccess>(
        static_cast<ExprAggregateDataAccess *>(expr.release())));
  };

  ExprAggregateDataAccess(const source::Node &srcNode, BaseExpr base,
                          const Symbol &sym, const unsigned accessIdx)
      : Expr(Kind::ExprAggregateDataAccess, srcNode, sym.getType()),
        Base(std::move(base)), AccessIdx(accessIdx), FieldAccessedSym(&sym) {}

  ExprAggregateDataAccess(const source::Node &srcNode, BaseExpr base,
                          const Type &ty, const unsigned accessIdx)
      : Expr(Kind::ExprAggregateDataAccess, srcNode, ty), Base(std::move(base)),
        AccessIdx(accessIdx), FieldAccessedSym(nullptr) {}

  const Expr &getBase() const {
    if (auto *varRef = std::get_if<std::unique_ptr<ExprVarRef>>(&Base)) {
      return **varRef;
    }
    if (auto *tupleExpr = std::get_if<std::unique_ptr<ExprTuple>>(&Base)) {
      return **tupleExpr;
    }
    return *std::get<std::unique_ptr<ExprAggregateDataAccess>>(Base);
  }

  const Symbol *getBaseSymbol() const {
    assert(!std::holds_alternative<std::unique_ptr<ExprTuple>>(Base) &&
           "Cannot get symbol of tuple");
    if (auto *varRef = std::get_if<std::unique_ptr<ExprVarRef>>(&Base)) {
      return &varRef->get()->getSymbol();
    }
    return std::get<std::unique_ptr<ExprAggregateDataAccess>>(Base)
        ->getSymbol();
  }

  std::optional<llvm::StringRef> getName() const {
    return FieldAccessedSym ? FieldAccessedSym->getId()
                            : std::optional<llvm::StringRef>{};
  }
  unsigned getIdxAccess() const { return AccessIdx; }
  const Symbol *getSymbol() const { return FieldAccessedSym; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprAggregateDataAccess;
  }

private:
  BaseExpr Base;
  const unsigned AccessIdx;
  /// Symbol in case we are accessing a ExprVarRef
  const Symbol *FieldAccessedSym;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_EXPRS_H
