#include <tmplang/Tree/HIR/Decls.h>

using namespace tmplang::hir;

StringLiteral tmplang::hir::ToString(SubprogramDecl::FunctionKind kind) {
  switch (kind) {
  case SubprogramDecl::proc:
    return "proc";
  case SubprogramDecl::fn:
    return "fn";
  }
  llvm_unreachable("All cases covered");
}
