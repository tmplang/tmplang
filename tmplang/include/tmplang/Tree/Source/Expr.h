#ifndef TMPLANG_TREE_SOURCE_EXPR_H
#define TMPLANG_TREE_SOURCE_EXPR_H

#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/Node.h>

namespace tmplang::source {

class Expr : public Node {
public:
  virtual ~Expr() = default;

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprIntegerNumber;
  }

protected:
  explicit Expr(Node::Kind k) : Node(k) {}
};

class ExprStmt final : public Node {
public:
  explicit ExprStmt(std::unique_ptr<Expr> expr,
                    SpecificToken<TK_Semicolon> semiColon)
      : Node(Node::Kind::ExprStmt), E(std::move(expr)),
        Semicolon(std::move(semiColon)) {}

  const Expr *getExpr() const { return &*E; }
  const auto &getSemicolon() const { return Semicolon; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::ExprStmt;
  }

  tmplang::SourceLocation getBeginLoc() const override {
    return E ? E->getBeginLoc() : Semicolon.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Semicolon.getSpan().End;
  }

private:
  std::unique_ptr<Expr> E;
  SpecificToken<TK_Semicolon> Semicolon;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_EXPR_H
