#ifndef TMPLANG_TREE_SOURCE_DECLS_H
#define TMPLANG_TREE_SOURCE_DECLS_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/Decl.h>
#include <tmplang/Tree/Source/Types.h>

namespace tmplang::source {

class ParamDecl final : public Decl {
public:
  explicit ParamDecl(Token id, NamedType paramType)
      : Decl(Node::Kind::ParamDecl), ParamType(std::move(paramType)) {}
  virtual ~ParamDecl() = default;

  const NamedType &getType() const { return ParamType; }

  llvm::StringRef getName() const override { return Identifier.getLexeme(); }
  SourceLocation getBeginLoc() const override {
    return ParamType.getBeginLoc();
  }
  SourceLocation getEndLoc() const override { return ParamType.getEndLoc(); }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::ParamDecl;
  }

private:
  NamedType ParamType;
  Token Identifier;
};

class FunctionDecl final : public Decl {
public:
  struct ArrowAndType {
    Token Arrow;
    NamedType RetType;
  };

  struct ParamList {
    std::vector<source::ParamDecl> ParamList;
    /// Commas preceding after the first param. CommaList.size() must be equal
    /// to ParamList.size() - 1
    std::vector<Token> CommaList;
  };

  const NamedType *getReturnType() const {
    return OptArrowAndType ? &OptArrowAndType->RetType : nullptr;
  }
  const ParamList &getParams() const { return Params; }
  llvm::StringRef getName() const override { return Identifier.getLexeme(); }
  SourceLocation getBeginLoc() const override { return FuncType.StartLocation; }
  SourceLocation getEndLoc() const override { return RKeyBracket.EndLocation; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::FuncDecl;
  }

  /// fn foo: i32 a, f32 b -> i32 {}
  static FunctionDecl Create(Token funcType, Token identifier, Token colon,
                             ParamList paramList, ArrowAndType arrowAndType,
                             Token lKeyBracket, Token rKeyBracket);
  /// fn foo: i32 a, f32 b {}
  static FunctionDecl Create(Token funcType, Token identifier, Token colon,
                             ParamList paramList, Token lKeyBracket,
                             Token rKeyBracket);
  /// fn foo -> i32 {}
  static FunctionDecl Create(Token funcType, Token identifier,
                             ArrowAndType arrowAndType, Token lKeyBracket,
                             Token rKeyBracket);
  /// fn foo {}
  static FunctionDecl Create(Token funcType, Token identifier,
                             Token lKeyBracket, Token rKeyBracket);

  virtual ~FunctionDecl() = default;

private:
  FunctionDecl(Token funcType, Token identifier, llvm::Optional<Token> colon,
               ParamList paramList, llvm::Optional<ArrowAndType> arrowAndType,
               Token lKeyBracket, Token rKeyBracket);

  Token FuncType;
  Token Identifier;
  llvm::Optional<Token> Colon;
  ParamList Params;
  llvm::Optional<ArrowAndType> OptArrowAndType;
  Token LKeyBracket;
  /// TODO: Add body
  Token RKeyBracket;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_DECLS_H
