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
      return getDerived().traverseBuiltinType(*cast<BuiltinType>(&type));
    case Type::K_Tuple:
      return getDerived().traverseTupleType(*cast<TupleType>(&type));
      break;
    }
    llvm_unreachable("All cases are handled");
  }

  // TODO: Define nodes on .def file to use macro magic
  bool visitType(const Type &type) {
    switch (type.getKind()) {
    case Type::Kind::K_Builtin:
      return getDerived().visitBuiltinType(*cast<BuiltinType>(&type));
    case Type::K_Tuple:
      return getDerived().visitTupleType(*cast<TupleType>(&type));
      break;
    }
    llvm_unreachable("All cases are handled");
  }

protected:
  //=--------------------------------------------------------------------------=//
  // Begin visit functions
  //=--------------------------------------------------------------------------=//
  // TODO: Move all of these to .def file to use macro magic
  bool visitBuiltinType(const BuiltinType &) { return true; }
  bool visitTupleType(const TupleType &) { return true; }
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
  bool traverseTupleType(const TupleType &tupleType) {
    TRY_TO(visitType(tupleType));
    for (const Type *type : tupleType.getTypes()) {
      TRY_TO(traverseType(*type));
    }
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
