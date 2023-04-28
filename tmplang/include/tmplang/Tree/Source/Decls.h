#ifndef TMPLANG_TREE_SOURCE_DECLS_H
#define TMPLANG_TREE_SOURCE_DECLS_H

#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/CommonConstructs.h>
#include <tmplang/Tree/Source/Decl.h>
#include <tmplang/Tree/Source/Expr.h>
#include <tmplang/Tree/Source/Types.h>

namespace tmplang::source {

class ParamDecl final : public Decl, public TrailingOptComma {
public:
  explicit ParamDecl(RAIIType paramType, SpecificToken<TK_Identifier> id)
      : Decl(Kind::ParamDecl, std::move(id)), ParamType(std::move(paramType)) {}

  const Type &getType() const { return *ParamType; }

  tmplang::SourceLocation getBeginLoc() const override {
    return ParamType->getBeginLoc();
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Identifier.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::ParamDecl;
  }

private:
  RAIIType ParamType;
};

class SubprogramDecl final : public Decl {
public:
  struct ArrowAndType {
    SpecificToken<TK_RArrow> Arrow;
    RAIIType RetType;
  };

  struct Block {
    SpecificToken<TK_LKeyBracket> LKeyBracket;
    std::vector<source::ExprStmt> Exprs;
    SpecificToken<TK_RKeyBracket> RKeyBracket;
  };

  const Type *getReturnType() const {
    return OptArrowAndType ? OptArrowAndType->RetType.get() : nullptr;
  }
  const ArrayRef<ParamDecl> getParams() const { return ParamList.Items; }
  const auto &getFuncType() const { return FuncType; }
  const std::optional<SpecificToken<TK_Colon>> &getColon() const {
    return Colon;
  }
  const auto &getLKeyBracket() const { return B.LKeyBracket; }
  const auto &getRKeyBracket() const { return B.RKeyBracket; }
  const Block &getBlock() const { return B; }
  std::optional<SpecificToken<TK_RArrow>> getArrow() const {
    return OptArrowAndType ? OptArrowAndType->Arrow
                           : std::optional<SpecificToken<TK_RArrow>>{};
  }

  tmplang::SourceLocation getBeginLoc() const override {
    return FuncType.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return B.RKeyBracket.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::SubprogramDecl;
  }

  /// fn foo: i32 a, f32 b -> i32 {}
  /// fn foo: i32 a, f32 b {}
  /// fn foo -> i32 {}
  /// fn foo {}
  SubprogramDecl(SpecificToken<TK_FnType, TK_ProcType> funcType,
                 SpecificToken<TK_Identifier> identifier, Block block,
                 std::optional<SpecificToken<TK_Colon>> colon,
                 source::VariadicList<source::ParamDecl> paramList,
                 std::optional<ArrowAndType> arrowAndType = nullopt)
      : Decl(Kind::SubprogramDecl, std::move(identifier)),
        FuncType(std::move(funcType)), Colon(std::move(colon)),
        ParamList(std::move(paramList)),
        OptArrowAndType(std::move(arrowAndType)), B(std::move(block)) {}

  SubprogramDecl(SubprogramDecl &&) = default;
  SubprogramDecl &operator=(SubprogramDecl &&) = default;
  virtual ~SubprogramDecl() = default;

private:
  SpecificToken<TK_FnType, TK_ProcType> FuncType;
  std::optional<SpecificToken<TK_Colon>> Colon;
  source::VariadicList<source::ParamDecl> ParamList;
  std::optional<ArrowAndType> OptArrowAndType;
  Block B;
};

class DataFieldDecl final : public Decl, public TrailingOptComma {
public:
  const auto &getColon() const { return Colon; }
  const Type &getType() const { return *Ty; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Identifier.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return getComma() ? getComma()->getSpan().End : Ty->getEndLoc();
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::DataFieldDecl;
  }

  DataFieldDecl(SpecificToken<TK_Identifier> id, SpecificToken<TK_Colon> colon,
                RAIIType ty)
      : Decl(Node::Kind::DataFieldDecl, std::move(id)), Colon(std::move(colon)),
        Ty(std::move(ty)) {
    assert(Ty);
  }

private:
  SpecificToken<TK_Colon> Colon;
  RAIIType Ty;
};

class DataDecl final : public Decl {
public:
  const auto &getDataKeyword() const { return DataKeyword; }
  llvm::ArrayRef<DataFieldDecl> getFields() const { return Fields.Items; }
  const auto &getStartingEq() const { return StartingEq; }
  const auto &getEndingSemicolon() const { return EndingSemicolon; }

  tmplang::SourceLocation getBeginLoc() const override {
    return DataKeyword.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return EndingSemicolon.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::DataDecl;
  }

  DataDecl(SpecificToken<TK_Data> dataKeyword, SpecificToken<TK_Identifier> id,
           SpecificToken<TK_Eq> startingEq,
           OneElementOrMoreList<DataFieldDecl> fields,
           SpecificToken<TK_Semicolon> endingSemicolon)
      : Decl(Kind::DataDecl, std::move(id)),
        DataKeyword(std::move(dataKeyword)), StartingEq(std::move(startingEq)),
        Fields(std::move(fields)), EndingSemicolon(std::move(endingSemicolon)) {
  }

  virtual ~DataDecl() = default;

private:
  SpecificToken<TK_Data> DataKeyword;
  SpecificToken<TK_Eq> StartingEq;
  OneElementOrMoreList<DataFieldDecl> Fields;
  SpecificToken<TK_Semicolon> EndingSemicolon;
};

class UnionAlternativeFieldDecl final : public Decl, public TrailingOptComma {
public:
  const auto &getColon() const { return Colon; }
  const Type &getType() const { return *Ty; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Identifier.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return getComma() ? getComma()->getSpan().End : Ty->getEndLoc();
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionAlternativeFieldDecl;
  }

  UnionAlternativeFieldDecl(SpecificToken<TK_Identifier> id,
                            SpecificToken<TK_Colon> colon, RAIIType ty)
      : Decl(Kind::UnionAlternativeFieldDecl, std::move(id)),
        Colon(std::move(colon)), Ty(std::move(ty)) {
    assert(Ty);
  }

private:
  SpecificToken<TK_Colon> Colon;
  RAIIType Ty;
};

class UnionAlternativeDecl final : public Decl, public TrailingOptComma {
public:
  const auto &getLhsParen() const { return LhsParen; }
  ArrayRef<UnionAlternativeFieldDecl> getFields() const { return Fields.Items; }
  const auto &getRhsParen() const { return RhsParen; }

  tmplang::SourceLocation getBeginLoc() const override {
    return Identifier.getSpan().Start;
  }

  tmplang::SourceLocation getEndLoc() const override {
    return getComma() ? getComma()->getSpan().End : RhsParen.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionAlternativeDecl;
  }

  UnionAlternativeDecl(SpecificToken<TK_Identifier> id,
                       SpecificToken<TK_LParentheses> lParen,
                       OneElementOrMoreList<UnionAlternativeFieldDecl> fields,
                       SpecificToken<TK_RParentheses> rParen)
      : Decl(Kind::UnionAlternativeDecl, std::move(id)),
        LhsParen(std::move(lParen)), Fields(std::move(fields)),
        RhsParen(std::move(rParen)) {}

private:
  SpecificToken<TK_LParentheses> LhsParen;
  OneElementOrMoreList<UnionAlternativeFieldDecl> Fields;
  SpecificToken<TK_RParentheses> RhsParen;
};

class UnionDecl final : public Decl {
public:
  const auto &getUnionKeyword() const { return UnionKeyword; }
  const auto &getStartingEq() const { return StartingEq; }
  llvm::ArrayRef<UnionAlternativeDecl> getAlternatives() const {
    return Alternatives.Items;
  }
  const auto &getEndingSemicolon() const { return EndingSemicolon; }

  tmplang::SourceLocation getBeginLoc() const override {
    return UnionKeyword.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return EndingSemicolon.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionDecl;
  }

  UnionDecl(SpecificToken<TK_Union> unionKeyword,
            SpecificToken<TK_Identifier> id, SpecificToken<TK_Eq> startingEq,
            OneElementOrMoreList<UnionAlternativeDecl> alternatives,
            SpecificToken<TK_Semicolon> endingSemicolon)
      : Decl(Kind::UnionDecl, std::move(id)),
        UnionKeyword(std::move(unionKeyword)),
        StartingEq(std::move(startingEq)),
        Alternatives(std::move(alternatives)),
        EndingSemicolon(std::move(endingSemicolon)) {}

  virtual ~UnionDecl() = default;

private:
  SpecificToken<TK_Union> UnionKeyword;
  SpecificToken<TK_Eq> StartingEq;
  OneElementOrMoreList<UnionAlternativeDecl> Alternatives;
  SpecificToken<TK_Semicolon> EndingSemicolon;
};

class PlaceholderDecl final : public Decl {
public:
  PlaceholderDecl(SpecificToken<TK_Identifier> id)
      : Decl(Kind::PlaceholderDecl, std::move(id)) {}

  tmplang::SourceLocation getBeginLoc() const override {
    return Identifier.getSpan().Start;
  }
  tmplang::SourceLocation getEndLoc() const override {
    return Identifier.getSpan().End;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::PlaceholderDecl;
  }
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_DECLS_H
