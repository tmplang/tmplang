#ifndef TMPLANG_TREE_HIR_RECURSIVENODEVISITOR_H
#define TMPLANG_TREE_HIR_RECURSIVENODEVISITOR_H

#include <llvm/Support/Casting.h>
#include <tmplang/Tree/HIR/CompilationUnit.h>
#include <tmplang/Tree/HIR/Decls.h>
#include <tmplang/Tree/HIR/Exprs.h>

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
    for (const auto &decl : compilationUnit.getTopLevelDecls()) {
      TRY_TO(traverseNode(*decl));
    }
    return true;
  }
  bool traverseSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    TRY_TO(visitNode(subprogramDecl));
    for (const auto &param : subprogramDecl.getParams()) {
      TRY_TO(traverseNode(param));
    }
    for (const auto &expr : subprogramDecl.getBody()) {
      TRY_TO(traverseNode(*expr));
    }
    return true;
  }
  bool traverseDataDecl(const DataDecl &dataDecl) {
    TRY_TO(visitNode(dataDecl));
    for (const auto &field : dataDecl.getFields()) {
      TRY_TO(traverseNode(field));
    }
    return true;
  }
  bool traverseDataFieldDecl(const DataFieldDecl &dataFieldDecl) {
    TRY_TO(visitNode(dataFieldDecl));
    return true;
  }
  bool traverseParamDecl(const ParamDecl &paramDecl) {
    TRY_TO(visitNode(paramDecl));
    return true;
  }
  bool traverseExprIntegerNumber(const ExprIntegerNumber &num) {
    TRY_TO(visitNode(num));
    return true;
  }
  bool traverseExprTuple(const ExprTuple &exprTuple) {
    TRY_TO(visitNode(exprTuple));
    for (const auto &tupleVal : exprTuple.getVals()) {
      TRY_TO(traverseNode(*tupleVal));
    }
    return true;
  }
  bool traverseExprRet(const ExprRet &exprRet) {
    TRY_TO(visitNode(exprRet));
    if (auto *retExpr = exprRet.getReturnedExpr()) {
      TRY_TO(traverseNode(*retExpr));
    }
    return true;
  }
  bool traverseExprVarRef(const ExprVarRef &exprVarRef) {
    TRY_TO(visitNode(exprVarRef));
    return true;
  }
  bool traverseExprAggregateDataAccess(
      const ExprAggregateDataAccess &exprDataFieldAcc) {
    TRY_TO(visitNode(exprDataFieldAcc));
    TRY_TO(traverseNode(exprDataFieldAcc.getBase()));
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
