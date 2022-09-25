#ifndef TMPLANG_AST_TYPES_H
#define TMPLANG_AST_TYPES_H

#include "llvm/ADT/StringRef.h"
#include <tmplang/AST/Type.h>

namespace tmplang {

class ASTContext;

class BuiltinType final : public Type {
public:
  enum Kind { K_Unit, K_i32 };

  Kind getBuiltinKind() const { return BKind; }

  static const BuiltinType &getType(const ASTContext &ASTCtxt, Kind type);

  static bool classof(const Type *T) {
    return T->getKind() == Type::Kind::K_Builtin;
  }

protected:
  friend class ASTContext;

  explicit BuiltinType(Kind kind) : Type(Type::Kind::K_Builtin), BKind(kind) {}
  virtual ~BuiltinType() = default;

  Kind BKind;
};

class NamedType final : public Type {
public:
  explicit NamedType(llvm::StringRef name)
      : Type(Type::Kind::K_Path), Name(name) {}

  llvm::StringRef getName() const { return Name; }

  static bool classof(const Type *T) {
    return T->getKind() == Type::Kind::K_Path;
  }

private:
  llvm::StringRef Name;
};

} // namespace tmplang

#endif // TMPLANG_AST_TYPES_H
