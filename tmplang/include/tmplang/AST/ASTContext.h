#ifndef TMPLANG_AST_ASTCONTEXT_H
#define TMPLANG_AST_ASTCONTEXT_H

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <llvm/ADT/StringMap.h>
#include <tmplang/AST/Types.h>

#include <memory>
#include <vector>

namespace tmplang {

class ASTContext : llvm::RefCountedBase<ASTContext> {
public:
  explicit ASTContext();
  ~ASTContext() = default;

  NamedType &getNamedType(llvm::StringRef name);

private:
  friend class BuiltinType;

  llvm::StringMap<std::unique_ptr<NamedType>> NamedTypes;
  /// NOTE: Once this grows, it can be a good idea to move it to  the heap
  ///       through PImpl idiom
  // Empty type
  BuiltinType UnitType;
  BuiltinType i32Type;
};

} // namespace tmplang

#endif // TMPLANG_AST_ASTCONTEXT_H
