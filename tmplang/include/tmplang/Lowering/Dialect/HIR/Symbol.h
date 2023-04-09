#ifndef TMPLANG_LOWERING_DIALECT_IR_SYMBOL_H
#define TMPLANG_LOWERING_DIALECT_IR_SYMBOL_H

#include <llvm/ADT/DenseMapInfo.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <tmplang/Lowering/Dialect/HIR/EnumAttrs.h>

namespace tmplang {

enum class SymbolVisibility { Public, Private };
llvm::StringLiteral ToString(SymbolVisibility);
std::optional<SymbolVisibility> ParseSymbolVisibility(mlir::StringAttr);
std::optional<SymbolVisibility> ParseSymbolVisibility(mlir::MLIRContext &ctx,
                                                      llvm::StringRef str);

/// Represent a space where symbols can be declared. Symbols are unique among
/// every SymbolTable
class SymbolTable {
public:
  using SymbolTableKey = std::pair<mlir::SymbolRefAttr, SymbolKind>;

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  mlir::Operation *lookup(SymbolTableKey) const;
  template <typename T> T lookup(SymbolTableKey key) const {
    return dyn_cast_or_null<T>(lookup(key));
  }
  bool insert(SymbolTableKey, mlir::Operation *);

  static SymbolAttr getSymbol(mlir::Operation *op);
  static void setSymbol(mlir::Operation *op, SymbolAttr sym);

  static std::optional<SymbolVisibility>
  getSymbolVisibility(mlir::Operation *op);
  static void setSymbolVisibility(mlir::Operation *op, SymbolVisibility vis);

  static mlir::SymbolRefAttr getSymbolRef(mlir::Operation *op);
  static void setSymbolRef(mlir::Operation *op, mlir::SymbolRefAttr symRef);

  // Return the name of the attribute used for symbol
  static llvm::StringLiteral getSymbolAttrName() { return "symbol"; }

  /// Return the name of the attribute used for symbol visibility
  static llvm::StringLiteral getVisibilityAttrName() {
    return "symbol_visibility";
  }

  static llvm::StringLiteral getSymbolRefAttrName() { return "symbol_ref"; }

  /// Returns the nearest symbol table from a given operation `from`. Returns
  /// nullptr if no valid parent symbol table could be found.
  static mlir::Operation *getNearestSymbolTable(mlir::Operation *from);

  /// Returns the operation registered with the given symbol from (including)
  /// the Operation upwards on the Operation's with the 'OpTrait::SymbolTable'
  /// trait. Returns nullptr if no valid symbol was found.
  static mlir::Operation *lookupSymbolFrom(mlir::Operation *, SymbolTableKey);
  template <typename T>
  static T lookupSymbolFrom(mlir::Operation *from, SymbolTableKey key) {
    return dyn_cast_or_null<T>(lookupSymbolFrom(from, key));
  }

  /// Returns the operation registered with the given symbol name within the
  /// closest parent operation of, or including, 'from' with the
  /// 'OpTrait::SymbolTable' trait. Returns nullptr if no valid symbol was
  /// found.
  static mlir::Operation *lookupNearestSymbolFrom(mlir::Operation *,
                                                  SymbolTableKey);
  template <typename T>
  static T lookupNearestSymbolFrom(mlir::Operation *from, SymbolTableKey key) {
    return dyn_cast_or_null<T>(lookupNearestSymbolFrom(from, key));
  }

  /// Returns the operation registered with the given symbol name with the
  /// regions of 'symbolTableOp'. 'symbolTableOp' is required to be an operation
  /// with the 'OpTrait::SymbolTable' trait.
  static mlir::Operation *lookupSymbolIn(mlir::Operation *op, SymbolTableKey);

private:
  /// This is a mapping from a name to the symbol with that name. They key is
  /// always known to be a StringAttr.
  llvm::DenseMap<SymbolTableKey, mlir::Operation *> Table;
};

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes. This avoids them being template
// instantiated/duplicated.
namespace details {
mlir::LogicalResult verifySymbolUser(mlir::Operation *op);
mlir::LogicalResult verifySymbol(mlir::Operation *op);
mlir::LogicalResult verifySymbolTable(mlir::Operation *op);
} // namespace details

namespace OpTrait {

template <typename ConcreteType>
class SymbolUser : public mlir::OpTrait::TraitBase<ConcreteType, SymbolUser> {
public:
  static mlir::LogicalResult verifyRegionTrait(mlir::Operation *op) {
    return details::verifySymbolUser(op);
  }
};

template <typename ConcreteType>
class Symbol : public mlir::OpTrait::TraitBase<ConcreteType, Symbol> {
public:
  static mlir::LogicalResult verifyRegionTrait(mlir::Operation *op) {
    return details::verifySymbol(op);
  }
};

template <typename ConcreteType>
class SymbolTable : public mlir::OpTrait::TraitBase<ConcreteType, SymbolTable> {
public:
  static mlir::LogicalResult verifyRegionTrait(mlir::Operation *op) {
    return details::verifySymbolTable(op);
  }
};

} // namespace OpTrait

} // namespace tmplang

// Include the interfaces
#include <tmplang/Lowering/Dialect/HIR/TmplangSymbol.h.inc>

#endif // TMPLANG_LOWERING_DIALECT_IR_SYMBOL_H
