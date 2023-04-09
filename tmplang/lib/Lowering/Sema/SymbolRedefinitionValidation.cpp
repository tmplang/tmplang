#include "SymbolRedefinitionValidation.h"

#include <mlir/IR/BuiltinTypes.h>
#include <tmplang/Lowering/Dialect/HIR/Ops.h>

using namespace tmplang;

bool tmplang::ValidateNoRedefinitionOfSymbols(tmplang::TranslationUnitOp tuOp) {
  bool success = true;

  mlir::DenseMap<mlir::Operation *, std::unique_ptr<SymbolTable>> opToSymTable;

  tuOp->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op->hasTrait<OpTrait::SymbolTable>()) {
      opToSymTable[op] = std::make_unique<SymbolTable>();
    }

    if (!op->hasTrait<OpTrait::Symbol>()) {
      return mlir::WalkResult::advance();
    }

    auto symbolAttr = SymbolTable::getSymbol(op);
    assert(symbolAttr);

    auto *symTableOp = SymbolTable::getNearestSymbolTable(op);
    assert(symTableOp && "All operations belong to a symTable");

    SymbolTable::SymbolTableKey key{symbolAttr.getName(),
                                    symbolAttr.getKind().getValue()};
    if (auto *entry = opToSymTable[symTableOp]->lookup(key)) {
      op->emitError()
          .append("Symbol '", key.first.getRootReference().getValue(),
                  "' re-declared.")
          .attachNote(entry->getLoc())
          .append("previously defined here");
      success = false;
    }

    opToSymTable[symTableOp]->insert(key, op);

    return mlir::WalkResult::advance();
  });

  return success;
}
