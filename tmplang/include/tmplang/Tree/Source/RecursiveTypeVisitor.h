#ifndef TMPLANG_TREE_SOURCE_RECURSIVETYPEVISITOR_H
#define TMPLANG_TREE_SOURCE_RECURSIVETYPEVISITOR_H

#include <llvm/Support/Casting.h>
#include <tmplang/Tree/Source/Types.h>

namespace tmplang::source {

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
    case Type::NamedType:
      return getDerived().traverseNamedType(*cast<NamedType>(&type));
    case Type::TupleType:
      return getDerived().traverseTupleType(*cast<TupleType>(&type));
      break;
    }
    llvm_unreachable("All cases are handled");
  }

  // TODO: Define nodes on .def file to use macro magic
  bool visitType(const Type &type) {
    switch (type.getKind()) {
    case Type::NamedType:
      return getDerived().visitNamedType(*cast<NamedType>(&type));
    case Type::TupleType:
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
  bool visitNamedType(const NamedType &) { return true; }
  bool visitTupleType(const TupleType &) { return true; }
  //=--------------------------------------------------------------------------=//
  // End visit functions
  //=--------------------------------------------------------------------------=//

  //=--------------------------------------------------------------------------=//
  // Begin recursive traversal functions
  //=--------------------------------------------------------------------------=//
  bool traverseNamedType(const NamedType &namedTy) {
    TRY_TO(visitType(namedTy));
    return true;
  }
  bool traverseTupleType(const TupleType &tupleType) {
    TRY_TO(visitType(tupleType));
    for (const RAIIType &type : tupleType.getTypes()) {
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

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_RECURSIVETYPEVISITOR_H
