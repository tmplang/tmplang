#ifndef TMPLANG_TREE_HIR_TYPES_H
#define TMPLANG_TREE_HIR_TYPES_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Tree/HIR/Type.h>

namespace tmplang::hir {

class HIRContext;

class BuiltinType final : public Type {
public:
  enum Kind { K_i32 };

  Kind getBuiltinKind() const { return BKind; }

  static const BuiltinType &get(const HIRContext &, Kind);
  static const BuiltinType *get(const HIRContext &, StringRef);

  static bool classof(const Type *T) {
    return T->getKind() == Type::Kind::K_Builtin;
  }

private:
  friend class HIRContext;
  explicit BuiltinType(Kind kind) : Type(Type::Kind::K_Builtin), BKind(kind) {}

  Kind BKind;
};

StringLiteral ToString(BuiltinType::Kind);

class TupleType final : public Type {
public:
  ArrayRef<const Type *> getTypes() const { return Types; }

  static const TupleType &get(HIRContext &, ArrayRef<const Type *> types);
  static const TupleType &getUnit(const HIRContext &);

  static bool classof(const Type *T) {
    return T->getKind() == Type::Kind::K_Tuple;
  }

private:
  friend class HIRContext;
  explicit TupleType(ArrayRef<const Type *> types)
      : Type(Type::Kind::K_Tuple), Types(types) {}

  SmallVector<const Type *> Types;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_TYPES_H
