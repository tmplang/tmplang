#ifndef TMPLANG_TREE_SOURCE_EXPRS_H
#define TMPLANG_TREE_SOURCE_EXPRS_H

#include <tmplang/Tree/Source/Expr.h>

#include <variant>

namespace tmplang::source {

class ExprIntegerNumber final : public Expr {
public:
  explicit ExprIntegerNumber(SpecificToken<TK_IntegerNumber> num)
      : Expr(Kind::ExprIntegerNumber), Number(num) {}

  const auto &getNumber() const { return Number; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Number.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Number.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprIntegerNumber;
  }

private:
  SpecificToken<TK_IntegerNumber> Number;
};

class TupleElem final : public Node {
public:
  TupleElem(std::unique_ptr<Expr> val)
      : Node(Kind::TupleElem), Val(std::move(val)) {}

  tmplang::SourceLocation getBeginLoc() const override {
    return Val->getBeginLoc();
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Comma ? Comma->getSpan().End : Val->getEndLoc();
  }

  const Expr &getVal() const { return *Val; }
  const Optional<SpecificToken<TK_Comma>> &getComma() const { return Comma; }
  void setComma(SpecificToken<TK_Comma> comma) { Comma = comma; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::TupleElem;
  }

private:
  std::unique_ptr<Expr> Val;
  Optional<SpecificToken<TK_Comma>> Comma;
};

class ExprTuple final : public Expr {
public:
  ExprTuple(SpecificToken<TK_LParentheses> lParen,
            SmallVector<TupleElem, 4> values,
            SpecificToken<TK_RParentheses> rParen)
      : Expr(Kind::ExprTuple), LParen(lParen), Values(std::move(values)),
        RParen(rParen) {}

  const auto &getLParen() const { return LParen; }
  const auto &getRParen() const { return RParen; }
  ArrayRef<TupleElem> getVals() const { return Values; }

  tmplang::SourceLocation getBeginLoc() const override {
    return LParen.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return RParen.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprTuple;
  }

private:
  SpecificToken<TK_LParentheses> LParen;
  SmallVector<TupleElem, 4> Values;
  SpecificToken<TK_RParentheses> RParen;
};

class ExprRet final : public Expr {
public:
  ExprRet(SpecificToken<TK_Ret> ret, std::unique_ptr<Expr> expr = nullptr)
      : Expr(Kind::ExprRet), Ret(ret), ExprToRet(std::move(expr)) {}

  const auto &getRetTk() const { return Ret; }
  const Expr *getReturnedExpr() const { return ExprToRet.get(); }

  tmplang::SourceLocation getBeginLoc() const override {
    return Ret.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return ExprToRet ? ExprToRet->getEndLoc() : Ret.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprRet;
  }

private:
  SpecificToken<TK_Ret> Ret;
  std::unique_ptr<Expr> ExprToRet;
};

class ExprVarRef final : public Expr {
public:
  ExprVarRef(SpecificToken<TK_Identifier> id)
      : Expr(Kind::ExprVarRef), Identifier(id) {}

  const auto &getIdentifier() const { return Identifier; }
  llvm::StringRef getName() const { return Identifier.getLexeme(); }

  tmplang::SourceLocation getBeginLoc() const override {
    return Identifier.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Identifier.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprVarRef;
  }

private:
  SpecificToken<TK_Identifier> Identifier;
};

class ExprAggregateDataAccess final : public Expr {
public:
  using BaseNode = std::variant<std::unique_ptr<ExprAggregateDataAccess>,
                                std::unique_ptr<source::ExprTuple>, ExprVarRef>;

  ExprAggregateDataAccess(BaseNode base, SpecificToken<TK_Dot> dot,
                          SpecificToken<TK_Identifier, TK_IntegerNumber> field)
      : Expr(Kind::ExprAggregateDataAccess), Base(std::move(base)), Dot(dot),
        Field(field) {}

  const Expr &getBase() const {
    if (auto *varRef = std::get_if<ExprVarRef>(&Base)) {
      return *varRef;
    }
    if (auto *tupleTy =
            std::get_if<std::unique_ptr<source::ExprTuple>>(&Base)) {
      return **tupleTy;
    }
    return *std::get<std::unique_ptr<ExprAggregateDataAccess>>(Base);
  }

  llvm::StringRef getBaseName() const {
    assert(!std::holds_alternative<std::unique_ptr<source::ExprTuple>>(Base) &&
           "Cannot get base name of tuple");
    if (auto *varRef = std::get_if<ExprVarRef>(&Base)) {
      return varRef->getName();
    }
    return std::get<std::unique_ptr<ExprAggregateDataAccess>>(Base)
        ->getBaseName();
  }

  const auto &getDot() const { return Dot; }
  const auto &getAccessedField() const { return Field; }

  llvm::StringRef getFieldName() const {
    assert(Field.is(TK_Identifier));
    return Field.getLexeme();
  }
  int32_t getNumber() const {
    assert(Field.is(TK_IntegerNumber));
    return Field.getNumber();
  }

  tmplang::SourceLocation getBeginLoc() const override {
    return getBase().getBeginLoc();
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Field.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprAggregateDataAccess;
  }

private:
  BaseNode Base;
  SpecificToken<TK_Dot> Dot;
  SpecificToken<TK_Identifier, TK_IntegerNumber> Field;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_EXPRS_H
