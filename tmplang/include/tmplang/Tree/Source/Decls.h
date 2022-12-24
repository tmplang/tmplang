#ifndef TMPLANG_TREE_SOURCE_DECLS_H
#define TMPLANG_TREE_SOURCE_DECLS_H

#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/CommonConstructs.h>
#include <tmplang/Tree/Source/Decl.h>
#include <tmplang/Tree/Source/Expr.h>
#include <tmplang/Tree/Source/Types.h>

namespace tmplang::source {

class ParamDecl final : public Decl {
public:
  explicit ParamDecl(RAIIType paramType, Token id)
      : Decl(Node::Kind::ParamDecl), ParamType(std::move(paramType)),
        Identifier(id) {}

  const Type &getType() const { return *ParamType; }

  StringRef getName() const override { return Identifier.getLexeme(); }
  Token getIdentifier() const { return Identifier; }
  const Optional<Token> &getComma() const { return Comma; }

  tmplang::SourceLocation getBeginLoc() const override {
    return ParamType->getBeginLoc();
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Identifier.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::ParamDecl;
  }

  void setComma(Token comma) { Comma = comma; }

private:
  RAIIType ParamType;
  Token Identifier;
  Optional<Token> Comma;
};

class FunctionDecl final : public Decl {
public:
  struct ArrowAndType {
    Token Arrow;
    RAIIType RetType;
  };

  struct Block {
    Token LKeyBracket;
    SmallVector<source::ExprStmt> Exprs;
    Token RKeyBracket;
  };

  const Type *getReturnType() const {
    return OptArrowAndType ? OptArrowAndType->RetType.get() : nullptr;
  }
  const ArrayRef<ParamDecl> getParams() const { return ParamList; }
  StringRef getName() const override { return Identifier.getLexeme(); }
  Token getFuncType() const { return FuncType; }
  Token getIdentifier() const { return Identifier; }
  const Optional<Token> &getColon() const { return Colon; }
  Optional<Token> getArrow() const {
    return OptArrowAndType ? OptArrowAndType->Arrow : Optional<Token>{};
  }
  Token getLKeyBracket() const { return B.LKeyBracket; }
  Token getRKeyBracket() const { return B.RKeyBracket; }
  const Block &getBlock() const { return B; }

  tmplang::SourceLocation getBeginLoc() const override {
    return FuncType.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return B.RKeyBracket.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::FunctionDecl;
  }

  /// fn foo: i32 a, f32 b -> i32 {}
  /// fn foo: i32 a, f32 b {}
  /// fn foo -> i32 {}
  /// fn foo {}
  FunctionDecl(Token funcType, Token identifier, Block block,
               Optional<Token> colon = llvm::None,
               SmallVector<source::ParamDecl, 4> paramList = {},
               Optional<ArrowAndType> arrowAndType = llvm::None)
      : Decl(Kind::FunctionDecl), FuncType(funcType), Identifier(identifier),
        Colon(colon), ParamList(std::move(paramList)),
        OptArrowAndType(std::move(arrowAndType)), B(std::move(block)) {}

  FunctionDecl(FunctionDecl &&) = default;
  FunctionDecl &operator=(FunctionDecl &&) = default;
  virtual ~FunctionDecl() = default;

private:
  Token FuncType;
  Token Identifier;
  Optional<Token> Colon;
  SmallVector<source::ParamDecl, 4> ParamList;
  Optional<ArrowAndType> OptArrowAndType;
  Block B;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_DECLS_H
