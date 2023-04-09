#include "SymbolTypeResolver.h"

#include <mlir/IR/BuiltinTypes.h>

using namespace tmplang;

namespace {

class SymbolTypeResolver {
public:
  SymbolTypeResolver(mlir::Operation *op) : Op(*op) {
    assert(op->hasTrait<OpTrait::Symbol>());
  }

  bool resolve() {
    auto symbolAttr = SymbolTable::getSymbol(&Op);
    assert(symbolAttr);

    auto resolvedTy = resolveType(symbolAttr.getType().getValue());
    if (!resolvedTy) {
      return false;
    }

    SymbolAttr resolvedSym =
        SymbolAttr::get(Op.getContext(), symbolAttr.getName(),
                        symbolAttr.getKind(), mlir::TypeAttr::get(resolvedTy));

    SymbolTable::setSymbol(&Op, resolvedSym);

    return true;
  }

private:
  mlir::Type resolveType(mlir::Type ty) {
    if (auto unresolvedTy = ty.dyn_cast<UnresolvedType>()) {
      return resolveType(unresolvedTy);
    }

    if (auto functionTy = ty.dyn_cast<mlir::FunctionType>()) {
      return resolveType(functionTy);
    }

    if (auto tupleTy = ty.dyn_cast<mlir::TupleType>()) {
      return resolveType(tupleTy);
    }

    if (auto dataTy = ty.dyn_cast<DataType>()) {
      return dataTy;
    }

    llvm_unreachable("Unexpected ty");
  }

  mlir::Type resolveType(mlir::FunctionType functionTy) {
    bool containsError = false;

    auto resolveAndMarkErrors = [&](mlir::Type paramTy) -> mlir::Type {
      auto resolvedParamTy = resolveType(paramTy);
      containsError |= !resolvedParamTy;
      return resolvedParamTy;
    };

    llvm::SmallVector<mlir::Type> paramTys;
    paramTys.reserve(functionTy.getInputs().size());
    transform(functionTy.getInputs(), std::back_inserter(paramTys),
              resolveAndMarkErrors);

    assert(functionTy.getResults().size() == 1);
    mlir::Type retTy = resolveAndMarkErrors(functionTy.getResults().front());

    if (containsError) {
      // Already reported
      return {};
    }

    return mlir::FunctionType::get(Op.getContext(), paramTys, retTy);
  }

  mlir::Type resolveType(UnresolvedType unresolvedTy) {
    auto symbolAttr = SymbolTable::getSymbol(&Op);
    assert(symbolAttr);

    SymbolTable::SymbolTableKey key{
        mlir::SymbolRefAttr::get(Op.getContext(), unresolvedTy.getName()),
        SymbolKind::Type};
    auto *defSymOp = SymbolTable::lookupSymbolFrom(&Op, key);
    if (!defSymOp) {
      // TODO: Change to use SourceModule so we can report with source code
      Op.emitError().append("Undefined type '", unresolvedTy.getName(), "'.");
      return {};
    }

    auto defSymOpAttr = SymbolTable::getSymbol(defSymOp);
    assert(defSymOpAttr);

    return defSymOpAttr.getType().getValue();
  }

  mlir::Type resolveType(mlir::TupleType tupleTy) {
    bool containsError = false;

    auto resolveAndMarkErrors = [&](mlir::Type paramTy) -> mlir::Type {
      auto resolvedParamTy = resolveType(paramTy);
      containsError |= !resolvedParamTy;
      return resolvedParamTy;
    };

    if (containsError) {
      // Already reported
      return {};
    }

    llvm::SmallVector<mlir::Type> tupleTys;
    tupleTys.reserve(tupleTy.getTypes().size());
    transform(tupleTy.getTypes(), std::back_inserter(tupleTys),
              resolveAndMarkErrors);

    return mlir::TupleType::get(Op.getContext(), tupleTys);
  }

private:
  mlir::Operation &Op;
};

} // namespace

bool tmplang::ResolveSymbolTypes(TranslationUnitOp tuOp) {
  bool success = true;

  tuOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (!op->hasTrait<OpTrait::Symbol>() || isa<BuiltinTypeOp>(op)) {
      // Skip ops without the symbol trait or builtins since they are already
      // resolved
      return mlir::WalkResult::advance();
    }

    success &= SymbolTypeResolver(op).resolve();

    return mlir::WalkResult::advance();
  });

  return success;
}
