#include <mlir/IR/Operation.h>
#include <tmplang/Lowering/Dialect/HIR/Traits.h>

#include <llvm/ADT/StringSwitch.h>
#include <tmplang/Lowering/Dialect/HIR/Ops.h>

using namespace tmplang;

llvm::StringLiteral tmplang::ToString(SymbolVisibility vis) {
  switch (vis) {
  case SymbolVisibility::Public:
    return "public";
  case SymbolVisibility::Private:
    return "private";
  }
  llvm_unreachable("All cases covered");
}

std::optional<SymbolVisibility>
tmplang::ParseSymbolVisibility(mlir::StringAttr str) {
  return llvm::StringSwitch<std::optional<SymbolVisibility>>(str.getValue())
      .Case("public", SymbolVisibility::Public)
      .Case("private", SymbolVisibility::Private)
      .Default(std::nullopt);
}

std::optional<SymbolVisibility>
tmplang::ParseSymbolVisibility(mlir::MLIRContext &ctx, llvm::StringRef str) {
  return ParseSymbolVisibility(mlir::StringAttr::get(&ctx, str));
}

static SymbolAttr getSymbolOfOp(mlir::Operation *op) {
  return op->getAttrOfType<SymbolAttr>(SymbolTable::getSymbolAttrName());
}

mlir::Operation *SymbolTable::lookup(SymbolTableKey key) const {
  return Table.lookup(key);
}

bool SymbolTable::insert(SymbolTableKey key, mlir::Operation *op) {
  return Table.insert({key, op}).second;
}

/*static*/ std::optional<SymbolVisibility>
SymbolTable::getSymbolVisibility(mlir::Operation *op) {
  mlir::StringAttr vis =
      op->getAttrOfType<mlir::StringAttr>(getVisibilityAttrName());
  return ParseSymbolVisibility(vis);
}

/*static*/ void SymbolTable::setSymbolVisibility(mlir::Operation *op,
                                                 SymbolVisibility vis) {
  op->setAttr(getVisibilityAttrName(),
              mlir::StringAttr::get(op->getContext(), ToString(vis)));
}

/*static*/ SymbolAttr SymbolTable::getSymbol(mlir::Operation *op) {
  return op->getAttrOfType<SymbolAttr>(getSymbolAttrName());
}

/*static*/ void SymbolTable::setSymbol(mlir::Operation *op, SymbolAttr kind) {
  op->setAttr(getSymbolAttrName(), kind);
}

/*static*/ mlir::SymbolRefAttr SymbolTable::getSymbolRef(mlir::Operation *op) {
  return op->getAttrOfType<mlir::SymbolRefAttr>(getSymbolRefAttrName());
}

/*static*/ void SymbolTable::setSymbolRef(mlir::Operation *op,
                                          mlir::SymbolRefAttr symRef) {
  op->setAttr(getSymbolRefAttrName(), symRef);
}

/*static*/ mlir::Operation *
SymbolTable::getNearestSymbolTable(mlir::Operation *from) {
  while (from) {
    if (from->hasTrait<OpTrait::SymbolTable>()) {
      return from;
    }
    from = from->getParentOp();
  }

  return nullptr;
}

/*static*/ mlir::Operation *SymbolTable::lookupSymbolIn(mlir::Operation *op,
                                                        SymbolTableKey key) {
  assert(op->hasTrait<OpTrait::SymbolTable>());
  mlir::Region &region = op->getRegion(0);
  if (region.empty()) {
    return nullptr;
  }

  for (auto &op : region.front()) {
    SymbolAttr symbol = getSymbolOfOp(&op);
    if (!symbol || symbol.getName() != key.first ||
        !bitEnumContainsAny(symbol.getKind().getValue(), key.second)) {
      continue;
    }
    return &op;
  }
  return nullptr;
}

/*static*/ mlir::Operation *SymbolTable::lookupSymbolFrom(mlir::Operation *from,
                                                          SymbolTableKey key) {
  from = getNearestSymbolTable(from);

  while (from) {
    if (auto *op = lookupSymbolIn(from, key)) {
      return op;
    }

    from = getNearestSymbolTable(from->getParentOp());
  }

  return nullptr;
}

/*static*/ mlir::Operation *
SymbolTable::lookupNearestSymbolFrom(mlir::Operation *from,
                                     SymbolTableKey key) {
  mlir::Operation *symbolTableOp = getNearestSymbolTable(from);
  return symbolTableOp ? lookupSymbolIn(symbolTableOp, key) : nullptr;
}

mlir::LogicalResult tmplang::details::verifySymbolUser(mlir::Operation *op) {
  return mlir::success(op->getAttrOfType<mlir::SymbolRefAttr>(
      SymbolTable::getSymbolRefAttrName()));
}

mlir::LogicalResult tmplang::details::verifySymbolTable(mlir::Operation *op) {
  return mlir::success();
}

mlir::LogicalResult tmplang::details::verifySymbol(mlir::Operation *op) {
  return mlir::success(SymbolTable::getSymbol(op));
}

#include <tmplang/Lowering/Dialect/HIR/TmplangSymbol.cpp.inc>
