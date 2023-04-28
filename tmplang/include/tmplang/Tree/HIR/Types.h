#ifndef TMPLANG_TREE_HIR_TYPES_H
#define TMPLANG_TREE_HIR_TYPES_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
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
  bool isUnit() const { return getTypes().empty(); }

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

class SubprogramType final : public Type {
public:
  const Type &getReturnType() const { return ReturnType; }
  ArrayRef<const Type *> getParamTypes() const { return ParamTypes; }

  static const SubprogramType &get(HIRContext &, const Type &retTy,
                                   ArrayRef<const Type *> paramTys);

  static bool classof(const Type *T) {
    return T->getKind() == Type::Kind::K_Subprogram;
  }

private:
  friend class HIRContext;
  explicit SubprogramType(const Type &retTy, ArrayRef<const Type *> paramTys)
      : Type(Type::Kind::K_Subprogram), ReturnType(retTy),
        ParamTypes(paramTys) {}

  const Type &ReturnType;
  SmallVector<const Type *> ParamTypes;
};

class DataType final : public Type {
public:
  ArrayRef<const Type *> getFieldsTypes() const { return ParamTypes; }
  llvm::StringRef getName() const { return Name; }

  static const DataType &get(HIRContext &, llvm::StringRef name,
                             ArrayRef<const Type *> paramTys);

  static bool classof(const Type *T) { return T->getKind() == K_Data; }

private:
  friend class HIRContext;
  explicit DataType(llvm::StringRef name, ArrayRef<const Type *> fieldsTys)
      : Type(K_Data), ParamTypes(fieldsTys), Name(name) {}

  SmallVector<const Type *> ParamTypes;
  SmallString<16> Name;
};

class UnionType final : public Type {
public:
  ArrayRef<const Type *> getAlternativeTypes() const { return AlternativeTys; }
  llvm::StringRef getName() const { return Name; }

  static const UnionType &get(HIRContext &, llvm::StringRef name,
                              ArrayRef<const Type *> alternativeTys);

  static bool classof(const Type *T) { return T->getKind() == K_Union; }

private:
  friend class HIRContext;
  explicit UnionType(llvm::StringRef name,
                     ArrayRef<const Type *> alternativeTys)
      : Type(K_Union), AlternativeTys(alternativeTys), Name(name) {}

  SmallVector<const Type *> AlternativeTys;
  SmallString<16> Name;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_TYPES_H
