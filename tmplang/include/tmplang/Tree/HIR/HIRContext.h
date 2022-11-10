#ifndef TMPLANG_TREE_HIR_HIRCONTEXT_H
#define TMPLANG_TREE_HIR_HIRCONTEXT_H

#include <llvm/ADT/IntrusiveRefCntPtr.h>
#include <tmplang/Tree/HIR/Types.h>

#include <deque>

namespace tmplang::hir {

class HIRContext : llvm::RefCountedBase<HIRContext> {
public:
  explicit HIRContext();
  ~HIRContext() = default;

private:
  friend class BuiltinType;
  friend class TupleType;

  /// NOTE: Once this grows, it can be a good idea to move it to  the heap
  ///       through PImpl idiom
  BuiltinType i32Type;
  TupleType UnitType;

  std::deque<TupleType> TupleTypes;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_HIRCONTEXT_H
