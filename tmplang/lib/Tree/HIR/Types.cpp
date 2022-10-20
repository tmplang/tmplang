#include <tmplang/Tree/HIR/Types.h>

#include <llvm/Support/ErrorHandling.h>
#include <tmplang/Tree/HIR/HIRContext.h>

using namespace tmplang::hir;

/*static*/ const BuiltinType &BuiltinType::getType(const HIRContext &astCtxt,
                                                   Kind kindToRetrieve) {
  switch (kindToRetrieve) {
  case BuiltinType::K_i32:
    return astCtxt.i32Type;
  }
  llvm_unreachable("BuiltinType case not covered!");
}
