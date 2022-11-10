#ifndef TMPLANG_TREE_SOURCE_TYPES_H
#define TMPLANG_TREE_SOURCE_TYPES_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/CommonConstructs.h>
#include <tmplang/Tree/Source/Type.h>

namespace tmplang::source {

class NamedType final : public Type {
public:
  explicit NamedType(Token identifier)
      : Type(Kind::NamedType), Identifier(identifier) {}

  llvm::StringRef getName() const { return Identifier.getLexeme(); }

  SourceLocation getBeginLoc() const override {
    return Identifier.StartLocation;
  }
  SourceLocation getEndLoc() const override { return Identifier.EndLocation; }

  static bool classof(const Type *T) { return T->getKind() == Kind::NamedType; }

protected:
  Token Identifier;
};

class TupleType final : public Type {
public:
  TupleType(Token lparentheses, llvm::SmallVectorImpl<RAIIType> &&types,
            llvm::SmallVectorImpl<Token> &&commas, Token rparentheses)
      : Type(Kind::TupleType), LParentheses(lparentheses),
        TypesAndCommas(std::move(types), std::move(commas)),
        RParentheses(rparentheses) {}

  llvm::ArrayRef<RAIIType> getTypes() const { return TypesAndCommas.Elems; }

  SourceLocation getBeginLoc() const override {
    return LParentheses.StartLocation;
  }
  SourceLocation getEndLoc() const override { return RParentheses.EndLocation; }

  static bool classof(const Type *T) { return T->getKind() == Kind::TupleType; }

private:
  Token LParentheses;
  CommaSeparatedList<RAIIType, 4> TypesAndCommas;
  Token RParentheses;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_TYPES_H
