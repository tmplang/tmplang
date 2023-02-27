#include <iterator>
#include <tmplang/Tree/HIR/Types.h>

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <tmplang/Tree/HIR/Decls.h>
#include <tmplang/Tree/HIR/HIRContext.h>

using namespace tmplang::hir;

/*static*/ const BuiltinType &BuiltinType::get(const HIRContext &ctx,
                                               Kind kindToRetrieve) {
  switch (kindToRetrieve) {
  case K_i32:
    return ctx.i32Type;
  }
  llvm_unreachable("BuiltinType case not covered!");
}

/*static*/ const BuiltinType *BuiltinType::get(const HIRContext &ctx,
                                               StringRef id) {
  return StringSwitch<const BuiltinType *>(id)
      .Case("i32", &ctx.i32Type)
      .Default(nullptr);
}

StringLiteral tmplang::hir::ToString(BuiltinType::Kind kind) {
  switch (kind) {
  case BuiltinType::K_i32:
    return "i32";
  }
  llvm_unreachable("All cases covered");
}

/*static*/ const TupleType &TupleType::get(HIRContext &ctx,
                                           ArrayRef<const Type *> types) {
  // FIXME: This is very naive and unoptimal, we could accelerate this using,
  //        for example, an index table by num of params

  // Search for it first in case already exists
  auto it = llvm::find_if(ctx.TupleTypes, [&](const TupleType &tupleTy) {
    return tupleTy.getTypes() == types;
  });

  return it != ctx.TupleTypes.end()
             ? *it
             : ctx.TupleTypes.emplace_back(TupleType(types));
}

/*static*/ const TupleType &TupleType::getUnit(const HIRContext &ctx) {
  return ctx.UnitType;
}

/*static*/ const SubprogramType &
SubprogramType::get(HIRContext &ctx, const Type &retTy,
                    ArrayRef<const Type *> paramTys) {
  // FIXME: This is very naive and unoptimal, we could accelerate this using,
  //        for example, an index table by tuple size

  // Search for it first in case already exists
  auto it = llvm::find_if(ctx.SubprogramTypes,
                          [&](const SubprogramType &subprogramTy) {
                            return subprogramTy.getParamTypes() == paramTys &&
                                   &subprogramTy.getReturnType() == &retTy;
                          });

  return it != ctx.SubprogramTypes.end() ? *it
                                         : ctx.SubprogramTypes.emplace_back(
                                               SubprogramType(retTy, paramTys));
}

/*static*/ const DataType &DataType::get(HIRContext &ctx, llvm::StringRef name,
                                         ArrayRef<const Type *> fieldsTys) {
  // FIXME: This is very naive and unoptimal, we could accelerate this using,
  //        for example, an index table by data size

  // Search for it first in case already exists
  auto it = llvm::find_if(ctx.DataTypes, [&](const DataType &dataType) {
    return dataType.getFieldsTypes() == fieldsTys && dataType.Name == name;
  });

  return it != ctx.DataTypes.end()
             ? *it
             : ctx.DataTypes.emplace_back(DataType(name, fieldsTys));
}

/*  static */ const UnionType &
UnionType::get(HIRContext &ctx, llvm::StringRef name,
               ArrayRef<const Type *> alternativeTys) {
  // FIXME: This is very naive and unoptimal, we could accelerate this using,
  //        for example, an index table by data size

  // Search for it first in case already exists
  auto it = llvm::find_if(ctx.UnionTypes, [&](const UnionType &unionTy) {
    return unionTy.getAlternativeTypes() == alternativeTys &&
           unionTy.Name == name;
  });

  return it != ctx.UnionTypes.end()
             ? *it
             : ctx.UnionTypes.emplace_back(UnionType(name, alternativeTys));
}
