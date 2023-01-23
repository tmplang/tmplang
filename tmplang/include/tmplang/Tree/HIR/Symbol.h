#ifndef TMPLANG_TREE_HIR_SYMBOL_H
#define TMPLANG_TREE_HIR_SYMBOL_H

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Allocator.h>

namespace tmplang::hir {

/// Foward declarations
class SymbolicScope;
class Symbol;
class Type;
class HIRContext;

enum class SymbolKind {
  /// Initial state of symbols that are unknown when queried. It is considered
  /// and error if after the resolving symbols pass any of these remains.
  Unresolved = 0,
  /// There are also two kinds of symbols. Those that are queryable from an
  /// expression or a type.
  Expr,
  Type
};
llvm::StringLiteral ToString(SymbolKind);

/// Represent a space where
class SymbolicScope {
public:
  /// All except the global scope has a containing scope
  const SymbolicScope &getContainingScope() const;
  bool isGlobalScope() const;

  bool containsSymbol(SymbolKind kind, llvm::StringRef id) const;
  [[maybe_unused]] Symbol &addSymbol(Symbol &);

  void dump(bool recursively = false) const;

  llvm::ArrayRef<std::reference_wrapper<Symbol>> getSymbols() const {
    return Symbols;
  }

private:
  /// Only the SymbolManager can build Scopes
  friend class SymbolManager;

  SymbolicScope() {}

private:
  /// TODO: Profile about a nice initial size for the array
  llvm::SmallVector<std::reference_wrapper<Symbol>> Symbols;
};

class Symbol {
public:
  SymbolKind getKind() const { return Kind; }
  llvm::StringRef getId() const { return Id; }
  const Type &getType() const { return Ty; }

  bool operator==(const Symbol &) const;

private:
  /// Only the SymbolManager can build Symbols
  friend class SymbolManager;

  /// Non-unresolvedSymbol constructor
  Symbol(const SymbolKind kind, llvm::StringRef id, const Type &ty)
      : Kind(kind), Id(id), Ty(ty) {}

  /// UnresolvedSymbol constructor
  /// It has unit type, althouhg it does not matter
  Symbol(const SymbolicScope &parentScope, const HIRContext &ctx);

private:
  /// The kind of this symbol
  const SymbolKind Kind;
  /// All symbols have an identifier
  llvm::StringRef Id;
  /// All symbols have a type
  const Type &Ty;
};

class SymbolManager {
public:
  SymbolManager(HIRContext &ctx)
      : GlobalScope(), UnresolvedSymbol(GlobalScope, ctx) {}

  SymbolicScope &getGlobalScope() { return GlobalScope; }
  const Symbol &getUnresolvedSymbol() const { return UnresolvedSymbol; }

  Symbol &createSymbol(const SymbolKind kind, llvm::StringRef id,
                       const Type &ty) {
    return *new (SymbolArena.Allocate()) Symbol(kind, id, ty);
  }

  SymbolicScope &createSymbolicScope() {
    return *new (SymbolicScopeArena.Allocate()) SymbolicScope();
  }

private:
  SymbolicScope GlobalScope;
  /// Symbol that has been not yet resolved. If any node still contains this
  /// node after resolving all symbols, means that there is a problem.
  const Symbol UnresolvedSymbol;

  /// Arenas that will store Symbols and Scopes
  llvm::SpecificBumpPtrAllocator<Symbol> SymbolArena;
  llvm::SpecificBumpPtrAllocator<SymbolicScope> SymbolicScopeArena;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_SYMBOL_H
