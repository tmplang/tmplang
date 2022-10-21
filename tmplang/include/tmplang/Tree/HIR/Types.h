#ifndef TMPLANG_TREE_HIR_TYPES_H
#define TMPLANG_TREE_HIR_TYPES_H

#include <llvm/ADT/StringRef.h>
#include <tmplang/Tree/HIR/Type.h>

namespace tmplang::hir {

class HIRContext;

class BuiltinType final : public Type {
public:
  enum Kind { K_i32, K_Unit };

  Kind getBuiltinKind() const { return BKind; }

  static const BuiltinType &get(const HIRContext &, Kind);
  static const BuiltinType *get(const HIRContext &, llvm::StringRef);

  static bool classof(const Type *T) {
    return T->getKind() == Type::Kind::K_Builtin;
  }

protected:
  friend class HIRContext;

  explicit BuiltinType(Kind kind) : Type(Type::Kind::K_Builtin), BKind(kind) {}
  virtual ~BuiltinType() = default;

  Kind BKind;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_TYPES_H
