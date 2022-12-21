#ifndef TMPLANG_TREE_SOURCE_EXPRS_H
#define TMPLANG_TREE_SOURCE_EXPRS_H

#include <tmplang/Tree/Source/Expr.h>

namespace tmplang::source {

class ExprIntegerNumber final : public Expr {
public:
  explicit ExprIntegerNumber(Token num) : Expr(Kind::ExprIntegerNumber), Number(num) {}

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

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_EXPRS_H
