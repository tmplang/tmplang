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
    case Type::K_Subprogram:
      return getDerived().traverseSubprogramType(*cast<SubprogramType>(&type));
    case Type::K_Data:
      return getDerived().traverseDataType(*cast<DataType>(&type));
    case Type::K_Union:
      return getDerived().traverseUnionType(*cast<UnionType>(&type));
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
    case Type::K_Subprogram:
      return getDerived().visitSubprogramType(*cast<SubprogramType>(&type));
    case Type::K_Data:
      return getDerived().visitDataType(*cast<DataType>(&type));
    case Type::K_Union:
      return getDerived().visitUnionType(*cast<UnionType>(&type));
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
  bool visitSubprogramType(const SubprogramType &) { return true; }
  bool visitDataType(const DataType &) { return true; }
  bool visitUnionType(const UnionType &) { return true; }
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
  bool traverseSubprogramType(const SubprogramType &subprogramTy) {
    TRY_TO(visitType(subprogramTy));
    for (const Type *type : subprogramTy.getParamTypes()) {
      TRY_TO(traverseType(*type));
    }
    TRY_TO(traverseType(subprogramTy.getReturnType()));
    return true;
  }
  bool traverseDataType(const DataType &dataTy) {
    TRY_TO(visitType(dataTy));
    for (const Type *type : dataTy.getFieldsTypes()) {
      TRY_TO(traverseType(*type));
    }
    return true;
  }
  bool traverseUnionType(const UnionType &unionTy) {
    TRY_TO(visitType(unionTy));
    for (const Type *type : unionTy.getAlternativeTypes()) {
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
