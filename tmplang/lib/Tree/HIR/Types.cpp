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
  case K_Unit:
    return ctx.UnitType;
    break;
  }
  llvm_unreachable("BuiltinType case not covered!");
}

/*static*/ const BuiltinType *BuiltinType::get(const HIRContext &ctx,
                                               llvm::StringRef id) {
  return llvm::StringSwitch<const BuiltinType *>(id)
      .Case("i32", &ctx.i32Type)
      .Default(nullptr);
}

llvm::StringLiteral tmplang::hir::ToString(BuiltinType::Kind kind) {
  switch (kind) {
  case BuiltinType::K_i32:
    return "i32";
  case BuiltinType::K_Unit:
    return "()";
  }
  llvm_unreachable("All cases covered");
}
