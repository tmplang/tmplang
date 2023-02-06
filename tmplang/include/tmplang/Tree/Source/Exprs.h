#ifndef TMPLANG_TREE_SOURCE_EXPRS_H
#define TMPLANG_TREE_SOURCE_EXPRS_H

#include <tmplang/Tree/Source/Expr.h>

#include <variant>

namespace tmplang::source {

class ExprIntegerNumber final : public Expr {
public:
  explicit ExprIntegerNumber(Token num)
      : Expr(Kind::ExprIntegerNumber), Number(num) {}

  Token getNumber() const { return Number; }

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
  Token Number;
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
  const Optional<Token> &getComma() const { return Comma; }
  void setComma(Token comma) { Comma = comma; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::TupleElem;
  }

private:
  std::unique_ptr<Expr> Val;
  Optional<Token> Comma;
};

class ExprTuple final : public Expr {
public:
  ExprTuple(Token lParen, SmallVector<TupleElem, 4> values, Token rParen)
      : Expr(Kind::ExprTuple), LParen(lParen), Values(std::move(values)),
        RParen(rParen) {}

  const Token &getLParen() const { return LParen; }
  const Token &getRParen() const { return RParen; }
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
  Token LParen;
  SmallVector<TupleElem, 4> Values;
  Token RParen;
};

class ExprRet final : public Expr {
public:
  ExprRet(Token ret, std::unique_ptr<Expr> expr = nullptr)
      : Expr(Kind::ExprRet), Ret(ret), ExprToRet(std::move(expr)) {}

  const Token &getRetTk() const { return Ret; }
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
  Token Ret;
  std::unique_ptr<Expr> ExprToRet;
};

class ExprVarRef final : public Expr {
public:
  ExprVarRef(Token id) : Expr(Kind::ExprVarRef), Identifier(id) {}

  const Token &getIdentifier() const { return Identifier; }
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
  Token Identifier;
};

class ExprAggregateDataAccess final : public Expr {
public:
  using BaseNode = std::variant<std::unique_ptr<ExprAggregateDataAccess>,
                                std::unique_ptr<source::ExprTuple>, ExprVarRef>;

  ExprAggregateDataAccess(BaseNode base, Token dot, Token field)
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

  Token getDot() const { return Dot; }
  Token getAccessedField() const { return Field; }
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
  Token Dot;
  SpecificToken<TK_Identifier, TK_IntegerNumber> Field;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_EXPRS_H
