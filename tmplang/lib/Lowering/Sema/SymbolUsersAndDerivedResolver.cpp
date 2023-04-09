#include "SymbolUsersAndDerivedResolver.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Visitors.h>
#include <tmplang/Lowering/Dialect/HIR/Ops.h>

using namespace tmplang;

static void
ResolveAndReplace(mlir::Operation *op,
                  std::function<mlir::Operation *(mlir::OpBuilder &)>
                      resolvedOpFuncGenerator) {
  mlir::OpBuilder builder(op->getContext());
  mlir::OperationState state(op->getLoc(), op->getName());

  builder.setInsertionPoint(op);
  auto *resolvedOp = resolvedOpFuncGenerator(builder);
  op->replaceAllUsesWith(resolvedOp);
  op->erase();
}

static bool ResolveTupleAccessOp(TupleAccessOp tupleOp) {
  auto tupleType = tupleOp.getInput().getType().dyn_cast<mlir::TupleType>();
  if (!tupleType) {
    tupleOp->emitError().append("Accessing with idx '",
                                tupleOp.getIdx().getZExtValue(),
                                "' to non tuple type");
    return false;
  }

  if (tupleOp.getIdx().getZExtValue() >= tupleType.getTypes().size()) {
    tupleOp->emitError().append("Out of range tuple access with index '",
                                tupleOp.getIdx().getZExtValue(),
                                "' on tuple with '",
                                tupleType.getTypes().size(), "' types");
    return false;
  }

  ResolveAndReplace(tupleOp, [&](mlir::OpBuilder &builder) {
    return builder.create<TupleAccessOp>(
        tupleOp->getLoc(), tupleType.getType(tupleOp.getIdx().getZExtValue()),
        tupleOp.getInput(), tupleOp.getIdx());
  });

  return true;
}

static bool ResolveDataAccessOp(DataAccessOp dataAccessOp) {
  auto dataType = dataAccessOp.getInput().getType().dyn_cast<DataType>();
  if (!dataType) {
    dataAccessOp->emitError().append("Accessing with name '",
                                     dataAccessOp.getSymbolRef(),
                                     "' to non data type");
    return false;
  }

  SymbolTable::SymbolTableKey key{
      mlir::SymbolRefAttr::get(dataAccessOp->getContext(), dataType.getName()),
      SymbolKind::DataType};
  auto *defSymOp = SymbolTable::lookupSymbolFrom(dataAccessOp, key);
  assert(defSymOp && "If we got a data, it was previous validated on varRef");

  key = SymbolTable::SymbolTableKey{dataAccessOp.getSymbolRef(),
                                    SymbolKind::DataField};
  auto *field = SymbolTable::lookupSymbolIn(defSymOp, key);
  if (!field) {
    dataAccessOp->emitError().append("No field in type '", dataType.getName(),
                                     "' named '", dataAccessOp.getSymbolRef(),
                                     "'");
    return false;
  }

  auto symbolAttr = SymbolTable::getSymbol(field);
  assert(symbolAttr);

  auto symbolRefAttr = SymbolTable::getSymbolRef(dataAccessOp);
  assert(symbolRefAttr);

  ResolveAndReplace(dataAccessOp, [&](mlir::OpBuilder &builder) {
    return builder.create<DataAccessOp>(dataAccessOp->getLoc(),
                                        symbolAttr.getType().getValue(),
                                        dataAccessOp.getInput(), symbolRefAttr);
  });

  return true;
}

static bool ResolveSymbolUserOp(mlir::Operation *op) {
  auto symbolRefAttr = SymbolTable::getSymbolRef(op);
  assert(symbolRefAttr);

  SymbolTable::SymbolTableKey key{symbolRefAttr, SymbolKind::VarDecl};
  auto *defSymOp = SymbolTable::lookupSymbolFrom(op, key);
  if (!defSymOp) {
    op->emitError().append("Referencing undefined symbol '", symbolRefAttr,
                           "'");
    return false;
  }
  assert(defSymOp->hasTrait<OpTrait::Symbol>());

  auto symbolAttr = SymbolTable::getSymbol(defSymOp);
  assert(symbolAttr);

  ResolveAndReplace(op, [&](mlir::OpBuilder &builder) {
    return builder.create<VarRefOp>(
        op->getLoc(), symbolAttr.getType().getValue(), symbolRefAttr);
  });

  return true;
}

bool tmplang::ResolveSymbolsUsersAndDerived(tmplang::TranslationUnitOp tuOp) {
  bool success = true;

  tuOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (auto tupleOp = dyn_cast<TupleAccessOp>(op)) {
      success |= ResolveTupleAccessOp(tupleOp);
      return mlir::WalkResult::skip();
    }

    if (!op->hasTrait<OpTrait::SymbolUser>()) {
      return mlir::WalkResult::advance();
    }

    if (auto dataAccessOp = dyn_cast<DataAccessOp>(op)) {
      success |= ResolveDataAccessOp(dataAccessOp);
      return mlir::WalkResult::skip();
    }

    success |= ResolveSymbolUserOp(op);
    return mlir::WalkResult::skip();
  });

  return success;
}
