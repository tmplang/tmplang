#ifndef TMPLANG_TREE_HIR_RECURSIVETYPEVISITOR_H
#define TMPLANG_TREE_HIR_RECURSIVETYPEVISITOR_H

#include <llvm/Support/Casting.h>
#include <tmplang/Tree/HIR/Types.h>

namespace tmplang::hir {

// Ugly repetitive control-flow code
#define TRY_TO(Call)                                                           \
  if (!getDerived().Call) {                                                    \
    return false;                                                              \
  }

template <typename Derived> class RecursiveTypeVisitor {
public:
  // TODO: Define nodes on .def file to use macro magic
  bool traverseType(const Type &type) {
    switch (type.getKind()) {
    case Type::Kind::K_Builtin:
      return getDerived().traverseBuiltinType(*llvm::cast<BuiltinType>(&type));
    }
    llvm_unreachable("All cases are handled");
  }

  // TODO: Define nodes on .def file to use macro magic
  bool visitType(const Type &type) {
    switch (type.getKind()) {
    case Type::Kind::K_Builtin:
      return getDerived().visitBuiltinType(*llvm::cast<BuiltinType>(&type));
    }
    llvm_unreachable("All cases are handled");
  }

protected:
  //=--------------------------------------------------------------------------=//
  // Begin visit functions
  //=--------------------------------------------------------------------------=//
  // TODO: Move all of these to .def file to use macro magic
  bool visitBuiltinType(const BuiltinType &) { return true; }
  //=--------------------------------------------------------------------------=//
  // End visit functions
  //=--------------------------------------------------------------------------=//

  //=--------------------------------------------------------------------------=//
  // Begin recursive traversal functions
  //=--------------------------------------------------------------------------=//
  bool traverseBuiltinType(const BuiltinType &builtinTy) {
    TRY_TO(visitType(builtinTy));
    return true;
  }
  //=--------------------------------------------------------------------------=//
  // End recursive traversal functions
  //=--------------------------------------------------------------------------=//
  Derived &getDerived() { return *static_cast<Derived *>(this); }
};

#undef TRY_TO

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_RECURSIVETYPEVISITOR_H
