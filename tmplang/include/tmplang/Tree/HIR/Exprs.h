#ifndef TMPLANG_TREE_HIR_EXPRS_H
#define TMPLANG_TREE_HIR_EXPRS_H

#include <llvm/ADT/APInt.h>
#include <tmplang/Tree/HIR/Decls.h>
#include <tmplang/Tree/HIR/Expr.h>
#include <tmplang/Tree/HIR/Symbol.h>
#include <tmplang/Tree/HIR/Types.h>
#include <tmplang/Tree/Source/Exprs.h>

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

class AggregateDestructuration;
class UnionDestructuration;

struct VoidPlaceholder {};
struct Otherwise {};

using ExprMatchCaseLhsVal =
    std::variant<std::unique_ptr<Expr>, PlaceholderDecl, VoidPlaceholder,
                 AggregateDestructuration, UnionDestructuration>;

class AggregateDestructurationElem : public Node {
public:
  AggregateDestructurationElem(
      const source::AggregateDestructurationElem &srcNode, const Type &elemTy,
      unsigned idx, std::unique_ptr<ExprMatchCaseLhsVal> value)
      : Node(Kind::AggregateDestructurationElem, srcNode), ElemTy(elemTy),
        IdxOfAggregateType(idx), Value(std::move(value)) {}

  static bool classof(const Node *node) {
    return node->getKind() == Kind::AggregateDestructurationElem;
  }

  const Type &getType() const { return ElemTy; }
  const ExprMatchCaseLhsVal &getValue() const;
  unsigned getIdxOfAggregateAccess() const { return IdxOfAggregateType; }

private:
  const Type &ElemTy;
  const unsigned IdxOfAggregateType;
  std::unique_ptr<ExprMatchCaseLhsVal> Value;
};

class AggregateDestructuration : public Node {
public:
  AggregateDestructuration(
      const source::TupleDestructuration &srcNode, const Type &typeDestructured,
      std::vector<AggregateDestructurationElem> &&destructuratedElems)
      : AggregateDestructuration(static_cast<const source::Node &>(srcNode),
                                 typeDestructured,
                                 std::move(destructuratedElems)) {}

  AggregateDestructuration(
      const source::DataDestructuration &srcNode, const Type &typeDestructured,
      std::vector<AggregateDestructurationElem> &&destructuratedElems)
      : AggregateDestructuration(static_cast<const source::Node &>(srcNode),
                                 typeDestructured,
                                 std::move(destructuratedElems)) {}

  const Type &getDestructuringType() const { return DestructuringTy; }

  ArrayRef<AggregateDestructurationElem> getElems() const {
    return DestructuratedElems;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::AggregateDestructuration;
  }

private:
  AggregateDestructuration(
      const source::Node &srcNode, const Type &typeDestructured,
      std::vector<AggregateDestructurationElem> &&destructuratedElems)
      : Node(Kind::AggregateDestructuration, srcNode),
        DestructuringTy(typeDestructured),
        DestructuratedElems(std::move(destructuratedElems)) {}

private:
  const Type &DestructuringTy;
  std::vector<AggregateDestructurationElem> DestructuratedElems;
};

class UnionDestructuration : public Node {
public:
  UnionDestructuration(const source::Node &srcNode, const Type &typeDes,
                       unsigned alternativeIdx,
                       AggregateDestructuration destructurated)
      : Node(Kind::UnionDestructuration, srcNode), DestructuringTy(typeDes),
        AlternativeIdx(alternativeIdx),
        Destructurated(std::move(destructurated)) {}

  unsigned getAlternativeIdx() const { return AlternativeIdx; }
  const Type &getDestructuringType() const { return DestructuringTy; }
  const AggregateDestructuration &getDestructuredData() const {
    return Destructurated;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionDestructuration;
  }

private:
  const Type &DestructuringTy;
  unsigned AlternativeIdx;
  AggregateDestructuration Destructurated;
};

class ExprMatchCase final : public Expr {
public:
  using LhsValue = std::variant<ExprMatchCaseLhsVal, Otherwise>;
  ExprMatchCase(const source::ExprMatchCase &srcNode, LhsValue matchingElem,
                std::unique_ptr<Expr> rhsExpr)
      : Expr(Kind::ExprMatchCase, srcNode, rhsExpr->getType()),
        Lhs(std::move(matchingElem)), Rhs(std::move(rhsExpr)) {}

  const LhsValue &getLhs() const { return Lhs; }
  const Expr &getRhs() const { return *Rhs; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprMatchCase;
  }

private:
  LhsValue Lhs;
  std::unique_ptr<Expr> Rhs;
};

class ExprMatch final : public Expr {
public:
  ExprMatch(const source::Node &srcNode, const Type &ty,
            std::unique_ptr<Expr> matchedExpr,
            SmallVectorImpl<std::unique_ptr<ExprMatchCase>> &&cases)
      : Expr(Kind::ExprMatch, srcNode, ty), MatchedExpr(std::move(matchedExpr)),
        ExprMatchCases(std::move(cases)) {}

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprMatch;
  }

  const Expr &getMatchedExpr() const { return *MatchedExpr; }

  ArrayRef<std::unique_ptr<ExprMatchCase>> getExprMatchCases() const {
    return ExprMatchCases;
  }

private:
  std::unique_ptr<Expr> MatchedExpr;
  // TODO: Profile nice initial number
  SmallVector<std::unique_ptr<ExprMatchCase>> ExprMatchCases;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_EXPRS_H
