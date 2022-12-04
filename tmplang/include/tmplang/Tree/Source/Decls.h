#ifndef TMPLANG_TREE_SOURCE_DECLS_H
#define TMPLANG_TREE_SOURCE_DECLS_H

#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/CommonConstructs.h>
#include <tmplang/Tree/Source/Decl.h>
#include <tmplang/Tree/Source/Types.h>

namespace tmplang::source {

class ParamDecl final : public Decl {
public:
  explicit ParamDecl(RAIIType paramType, Token id)
      : Decl(Node::Kind::ParamDecl), ParamType(std::move(paramType)),
        Identifier(id) {}

  const Type &getType() const { return *ParamType; }

  llvm::StringRef getName() const override { return Identifier.getLexeme(); }
  Token getIdentifier() const { return Identifier; }
  const llvm::Optional<Token> &getComma() const { return Comma; }

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
  llvm::Optional<Token> Comma;
};

class FunctionDecl final : public Decl {
public:
  struct ArrowAndType {
    Token Arrow;
    RAIIType RetType;
  };

  const Type *getReturnType() const {
    return OptArrowAndType ? OptArrowAndType->RetType.get() : nullptr;
  }
  const llvm::ArrayRef<ParamDecl> getParams() const { return ParamList; }
  llvm::StringRef getName() const override { return Identifier.getLexeme(); }
  Token getFuncType() const { return FuncType; }
  Token getIdentifier() const { return Identifier; }
  const llvm::Optional<Token> &getColon() const { return Colon; }
  llvm::Optional<Token> getArrow() const {
    return OptArrowAndType ? OptArrowAndType->Arrow : llvm::Optional<Token>{};
  }
  Token getLKeyBracket() const { return LKeyBracket; }
  Token getRKeyBracket() const { return RKeyBracket; }

  tmplang::SourceLocation getBeginLoc() const override {
    return FuncType.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return RKeyBracket.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::FuncDecl;
  }

  /// fn foo: i32 a, f32 b -> i32 {}
  /// fn foo: i32 a, f32 b {}
  /// fn foo -> i32 {}
  /// fn foo {}
  FunctionDecl(Token funcType, Token identifier, Token lKeyBracket,
               Token rKeyBracket, llvm::Optional<Token> colon = llvm::None,
               llvm::SmallVector<source::ParamDecl, 4> paramList = {},
               llvm::Optional<ArrowAndType> arrowAndType = llvm::None)
      : Decl(Kind::FuncDecl), FuncType(funcType), Identifier(identifier),
        Colon(colon), ParamList(std::move(paramList)),
        OptArrowAndType(std::move(arrowAndType)), LKeyBracket(lKeyBracket),
        RKeyBracket(rKeyBracket) {}

  FunctionDecl(FunctionDecl &&) = default;
  FunctionDecl &operator=(FunctionDecl &&) = default;
  virtual ~FunctionDecl() = default;

private:
  Token FuncType;
  Token Identifier;
  llvm::Optional<Token> Colon;
  llvm::SmallVector<source::ParamDecl, 4> ParamList;
  llvm::Optional<ArrowAndType> OptArrowAndType;
  Token LKeyBracket;
  /// TODO: Add body
  Token RKeyBracket;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_DECLS_H
