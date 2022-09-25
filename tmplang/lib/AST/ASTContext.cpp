#include "tmplang/AST/Types.h"
#include <tmplang/AST/ASTContext.h>

using namespace tmplang;

ASTContext::ASTContext()
    : UnitType(BuiltinType::Kind::K_Unit), i32Type(BuiltinType::Kind::K_i32) {}

NamedType &ASTContext::getNamedType(llvm::StringRef name) {
  auto it = NamedTypes.find(name);
  if (it != NamedTypes.end()) {
    return *it->getValue().get();
  }
  return *(NamedTypes[name] = std::make_unique<NamedType>(name));
}
