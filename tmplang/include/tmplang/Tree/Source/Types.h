#ifndef TMPLANG_TREE_SOURCE_TYPES_H
#define TMPLANG_TREE_SOURCE_TYPES_H

#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/Type.h>

namespace tmplang::source {

class NamedType final : public Type {
public:
  explicit NamedType(Token identifier)
      : Type(Kind::NamedType), Identifier(identifier) {}
  virtual ~NamedType() = default;

  llvm::StringRef getName() const { return Identifier.getLexeme(); }

  SourceLocation getBeginLoc() const override {
    return Identifier.StartLocation;
  }
  SourceLocation getEndLoc() const override { return Identifier.EndLocation; }

  static bool classof(const Type *T) { return T->getKind() == Kind::NamedType; }

protected:
  Token Identifier;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_TYPES_H
