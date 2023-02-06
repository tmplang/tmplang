#ifndef TMPLANG_TREE_SOURCE_TYPES_H
#define TMPLANG_TREE_SOURCE_TYPES_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/CommonConstructs.h>
#include <tmplang/Tree/Source/Type.h>

namespace tmplang::source {

class NamedType final : public Type {
public:
  explicit NamedType(SpecificToken<TK_Identifier> identifier)
      : Type(Kind::NamedType), Identifier(std::move(identifier)) {}

  StringRef getName() const { return Identifier.getLexeme(); }
  const auto &getIdentifier() const { return Identifier; }

  SourceLocation getBeginLoc() const override {
    return Identifier.getSpan().Start;
  }
  SourceLocation getEndLoc() const override { return Identifier.getSpan().End; }

  static bool classof(const Type *T) { return T->getKind() == Kind::NamedType; }

protected:
  SpecificToken<TK_Identifier> Identifier;
};

class TupleType final : public Type {
public:
  TupleType(SpecificToken<TK_LParentheses> lparentheses,
            SmallVectorImpl<RAIIType> &&types, SmallVectorImpl<Token> &&commas,
            SpecificToken<TK_RParentheses> rparentheses)
      : Type(Kind::TupleType), LParentheses(std::move(lparentheses)),
        TypesAndCommas(std::move(types), std::move(commas)),
        RParentheses(std::move(rparentheses)) {}

  const auto &getLParentheses() const { return LParentheses; }
  ArrayRef<RAIIType> getTypes() const { return TypesAndCommas.Elems; }
  ArrayRef<SpecificToken<TK_Comma>> getCommas() const {
    return TypesAndCommas.Commas;
  }
  const auto &getRParentheses() const { return RParentheses; }

  SourceLocation getBeginLoc() const override {
    return LParentheses.getSpan().Start;
  }
  SourceLocation getEndLoc() const override {
    return RParentheses.getSpan().End;
  }

  static bool classof(const Type *T) { return T->getKind() == Kind::TupleType; }

private:
  SpecificToken<TK_LParentheses> LParentheses;
  CommaSeparatedList<RAIIType, 4> TypesAndCommas;
  SpecificToken<TK_RParentheses> RParentheses;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_TYPES_H
