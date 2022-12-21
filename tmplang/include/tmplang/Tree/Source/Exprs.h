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

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_EXPRS_H
