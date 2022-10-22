#include <tmplang/Tree/HIR/Decls.h>

using namespace tmplang::hir;

llvm::StringLiteral tmplang::hir::ToString(FunctionDecl::FunctionKind kind) {
  switch (kind) {
  case FunctionDecl::proc:
    return "proc";
  case FunctionDecl::fn:
    return "fn";
  }
  llvm_unreachable("All cases covered");
}
