#include <tmplang/Tree/HIR/Decls.h>

#include <tmplang/Tree/HIR/Symbol.h>

using namespace tmplang::hir;

StringRef Decl::getName() const { return Sym.getId(); }
const Type &Decl::getType() const { return Sym.getType(); }

const SubprogramType &SubprogramDecl::getType() const {
  // All subprogram types contains subprogram type
  return *cast<hir::SubprogramType>(&this->getSymbol().getType());
}

StringLiteral tmplang::hir::ToString(SubprogramDecl::FunctionKind kind) {
  switch (kind) {
  case SubprogramDecl::proc:
    return "proc";
  case SubprogramDecl::fn:
    return "fn";
  }
  llvm_unreachable("All cases covered");
}
