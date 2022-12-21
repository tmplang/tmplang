#ifndef TMPLANG_TREE_HIR_RECURSIVENODEVISITOR_H
#define TMPLANG_TREE_HIR_RECURSIVENODEVISITOR_H

#include <llvm/Support/Casting.h>
#include <tmplang/Tree/HIR/CompilationUnit.h>
#include <tmplang/Tree/HIR/Decls.h>

namespace tmplang::hir {

// Ugly repetitive control-flow code
#define TRY_TO(Call)                                                           \
  if (!getDerived().Call) {                                                    \
    return false;                                                              \
  }

template <typename Derived> class RecursiveASTVisitor {
public:
  bool traverseNode(const Node &node) {
    switch (node.getKind()) {
#define HIRNode(K)                                                             \
  case Node::Kind::K:                                                          \
    return getDerived().traverse##K(*cast<K>(&node));
#include "../Nodes.def"
    }
    llvm_unreachable("All cases are handled");
  }

  bool visitNode(const Node &node) {
    switch (node.getKind()) {
#define HIRNode(K)                                                             \
  case Node::Kind::K:                                                          \
    return getDerived().visit##K(*cast<K>(&node));
#include "../Nodes.def"
    }
    llvm_unreachable("All cases are handled");
  }

protected:
  //=--------------------------------------------------------------------------=//
  // Begin visit functions
  //=--------------------------------------------------------------------------=//
#define HIRNode(K)                                                             \
  bool visit##K(const K &) { return true; }
#include "../Nodes.def"
  //=--------------------------------------------------------------------------=//
  // End visit functions
  //=--------------------------------------------------------------------------=//

  //=--------------------------------------------------------------------------=//
  // Begin recursive traversal functions
  //=--------------------------------------------------------------------------=//
  bool traverseCompilationUnit(const CompilationUnit &compilationUnit) {
    TRY_TO(visitNode(compilationUnit));
    for (const auto &function : compilationUnit.getFunctions()) {
      TRY_TO(traverseNode(function));
    }
    return true;
  }
  bool traverseFunctionDecl(const FunctionDecl &funcDecl) {
    TRY_TO(visitNode(funcDecl));
    for (const auto &param : funcDecl.getParams()) {
      TRY_TO(traverseNode(param));
    }
    return true;
  }
  bool traverseParamDecl(const ParamDecl &paramDecl) {
    TRY_TO(visitNode(paramDecl));
    return true;
  }
  //=--------------------------------------------------------------------------=//
  // End recursive traversal functions
  //=--------------------------------------------------------------------------=//

  Derived &getDerived() { return *static_cast<Derived *>(this); }
};

#undef TRY_TO

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_RECURSIVENODEVISITOR_H
