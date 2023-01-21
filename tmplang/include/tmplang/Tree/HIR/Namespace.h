#ifndef TMPLANG_TREE_HIR_NAMESPACE_H
#define TMPLANG_TREE_HIR_NAMESPACE_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Allocator.h>

namespace tmplang::hir {

/// Foward declarations
class Symbol;

/// Represents an accumulation of nested names. Allows querying for symbols by
/// name.
class Namespace {
private:
  /// Only the NamespaceManager can build Namespaces
  friend class NamespaceManager;

  /// Non-Global namespace constructor
  Namespace(llvm::StringRef name) : NamespaceName(name) {
    assert(!NamespaceName.empty() && "All namespaces must contain a name");
  }

  /// Global namespace constructor
  Namespace() {}

private:
  llvm::StringRef NamespaceName;
  // TODO: Profile the default number of symbols and namespaces
  llvm::SmallVector<const Symbol *, 6> ContainedSymbols;
  llvm::SmallVector<const Namespace *, 6> NestedNamespaces;
};

class NamespaceManager {
public:
  NamespaceManager() : GlobalNamespace() {}

  Namespace &getGlobalNamespace() { return GlobalNamespace; }

  Namespace &createNamespace(llvm::StringRef name) {
    return *new (NamespaceArena.Allocate()) Namespace(name);
  }

private:
  Namespace GlobalNamespace;
  /// Arena that will store all namespaces
  llvm::SpecificBumpPtrAllocator<Namespace> NamespaceArena;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_NAMESPACE_H
