#include <tmplang/Tree/HIR/Types.h>

#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/ErrorHandling.h>
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
  //        for example, an index table by tuple size

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
