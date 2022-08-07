#ifndef TMPLANG_AST_ASTCONTEXT_H
#define TMPLANG_AST_ASTCONTEXT_H

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <tmplang/AST/Types.h>

namespace tmplang {

class ASTContext : llvm::RefCountedBase<ASTContext> {
public:
  explicit ASTContext();
  ~ASTContext() = default;

private:
  friend class BuiltinType;

  /// NOTE: Once this grows, it can be a good idea to move it to  the heap
  ///       through PImpl idiom
  BuiltinType i32Type;
};

} // namespace tmplang

#endif // TMPLANG_AST_ASTCONTEXT_H
