#ifndef TMPLANG_TREE_SOURCE_RECURSIVENODEVISITOR_H
#define TMPLANG_TREE_SOURCE_RECURSIVENODEVISITOR_H

#include <tmplang/Tree/Source/CompilationUnit.h>
#include <tmplang/Tree/Source/Exprs.h>

namespace tmplang::source {

// Ugly repetitive control-flow code
#define TRY_TO(Call)                                                           \
  if (!getDerived().Call) {                                                    \
    return false;                                                              \
  }

template <typename Derived> class RecursiveASTVisitor {
public:
  bool traverseNode(const Node &node) {
    switch (node.getKind()) {
#define SourceNode(K)                                                          \
  case Node::Kind::K:                                                          \
    return getDerived().traverse##K(*cast<K>(&node));
#include "../Nodes.def"
    }
    llvm_unreachable("All cases are handled");
  }

  bool visitNode(const Node &node) {
    switch (node.getKind()) {
#define SourceNode(K)                                                          \
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
#define SourceNode(K)                                                          \
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
    for (const std::unique_ptr<Decl> &decl :
         compilationUnit.getTopLevelDecls()) {
      TRY_TO(traverseNode(*decl));
    }
    return true;
  }
  bool traverseSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    TRY_TO(visitNode(subprogramDecl));
    for (const ParamDecl &param : subprogramDecl.getParams()) {
      TRY_TO(traverseNode(param));
    }
    for (const ExprStmt &exprStmt : subprogramDecl.getBlock().Exprs) {
      TRY_TO(traverseNode(exprStmt));
    }
    return true;
  }
  bool traverseParamDecl(const ParamDecl &paramDecl) {
    TRY_TO(visitNode(paramDecl));
    return true;
  }
  bool traverseDataFieldDecl(const DataFieldDecl &dataFieldDecl) {
    TRY_TO(visitNode(dataFieldDecl));
    return true;
  }
  bool traverseDataDecl(const DataDecl &dataDecl) {
    TRY_TO(visitNode(dataDecl));
    for (const DataFieldDecl &dataFieldDecl : dataDecl.getFields()) {
      TRY_TO(traverseNode(dataFieldDecl));
    }
    return true;
  }
  bool traverseExprStmt(const ExprStmt &exprStmt) {
    TRY_TO(visitNode(exprStmt));
    if (auto *expr = exprStmt.getExpr()) {
      TRY_TO(traverseNode(*expr));
    }
    return true;
  }
  bool traverseExprIntegerNumber(const ExprIntegerNumber &exprIntegerNumber) {
    TRY_TO(visitNode(exprIntegerNumber));
    return true;
  }
  bool traverseExprTuple(const ExprTuple &exprTuple) {
    TRY_TO(visitNode(exprTuple));
    for (const auto &tupleVal : exprTuple.getVals()) {
      TRY_TO(traverseNode(tupleVal));
    }
    return true;
  }
  bool traverseTupleElem(const TupleElem &tupleElem) {
    TRY_TO(visitNode(tupleElem));
    TRY_TO(traverseNode(tupleElem.getVal()));
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
  //=--------------------------------------------------------------------------=//
  // End recursive traversal functions
  //=--------------------------------------------------------------------------=//

  Derived &getDerived() { return *static_cast<Derived *>(this); }
};

#undef TRY_TO

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_RECURSIVENODEVISITOR_H
