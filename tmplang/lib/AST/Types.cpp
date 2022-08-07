#include <tmplang/AST/Types.h>

#include <llvm/Support/ErrorHandling.h>
#include <tmplang/AST/ASTContext.h>

using namespace tmplang;

/*static*/ const BuiltinType &BuiltinType::getType(const ASTContext &astCtxt,
                                                   Kind kindToRetrieve) {
  switch (kindToRetrieve) {
  case BuiltinType::K_i32:
    return astCtxt.i32Type;
  }
  llvm_unreachable("BuiltinType case not covered!");
}
