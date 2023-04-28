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

  bool walkupNode(const Node &node) {
    switch (node.getKind()) {
#define HIRNode(K)                                                             \
  case Node::Kind::K:                                                          \
    return getDerived().walkup##K(*cast<K>(&node));
#include "../Nodes.def"
    }
    llvm_unreachable("All cases are handled");
  }

protected:
  //=--------------------------------------------------------------------------=//
  // Begin visit functions
  //=--------------------------------------------------------------------------=//
#define HIRNode(K)                                                             \
  bool visit##K(const K &) { return true; }                                    \
  bool walkup##K(const K &) { return true; }
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
    TRY_TO(walkupNode(compilationUnit));
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
    TRY_TO(walkupNode(subprogramDecl));
    return true;
  }
  bool traverseDataDecl(const DataDecl &dataDecl) {
    TRY_TO(visitNode(dataDecl));
    for (const auto &field : dataDecl.getFields()) {
      TRY_TO(traverseNode(field));
    }
    TRY_TO(walkupNode(dataDecl));
    return true;
  }
  bool traverseDataFieldDecl(const DataFieldDecl &dataFieldDecl) {
    TRY_TO(visitNode(dataFieldDecl));
    TRY_TO(walkupNode(dataFieldDecl));
    return true;
  }
  bool traverseParamDecl(const ParamDecl &paramDecl) {
    TRY_TO(visitNode(paramDecl));
    TRY_TO(walkupNode(paramDecl));
    return true;
  }
  bool traversePlaceholderDecl(const PlaceholderDecl &placeholderDecl) {
    TRY_TO(visitNode(placeholderDecl));
    TRY_TO(walkupNode(placeholderDecl));
    return true;
  }
  bool traverseExprIntegerNumber(const ExprIntegerNumber &num) {
    TRY_TO(visitNode(num));
    TRY_TO(walkupNode(num));
    return true;
  }
  bool traverseExprTuple(const ExprTuple &exprTuple) {
    TRY_TO(visitNode(exprTuple));
    for (const auto &tupleVal : exprTuple.getVals()) {
      TRY_TO(traverseNode(*tupleVal));
    }
    TRY_TO(walkupNode(exprTuple));
    return true;
  }
  bool traverseExprRet(const ExprRet &exprRet) {
    TRY_TO(visitNode(exprRet));
    if (auto *retExpr = exprRet.getReturnedExpr()) {
      TRY_TO(traverseNode(*retExpr));
    }
    TRY_TO(walkupNode(exprRet));
    return true;
  }
  bool traverseExprVarRef(const ExprVarRef &exprVarRef) {
    TRY_TO(visitNode(exprVarRef));
    TRY_TO(walkupNode(exprVarRef));
    return true;
  }
  bool traverseExprAggregateDataAccess(
      const ExprAggregateDataAccess &exprDataFieldAcc) {
    TRY_TO(visitNode(exprDataFieldAcc));
    TRY_TO(traverseNode(exprDataFieldAcc.getBase()));
    TRY_TO(walkupNode(exprDataFieldAcc));
    return true;
  }

  bool traverseExprMatch(const ExprMatch &exprMatch) {
    TRY_TO(visitNode(exprMatch));
    TRY_TO(traverseNode(exprMatch.getMatchedExpr()));
    for (const auto &matchCase : exprMatch.getExprMatchCases()) {
      TRY_TO(traverseNode(*matchCase));
    }
    TRY_TO(walkupNode(exprMatch));
    return true;
  }

  bool TraverseExprMatchCaseLhsVal(const ExprMatchCaseLhsVal &lhsVals) {
    return std::visit(
        source::visitors{
            [&](const std::unique_ptr<Expr> &expr) {
              TRY_TO(traverseNode(*expr));
              return true;
            },
            [&](const VoidPlaceholder &arg) { return true; },
            [&](const PlaceholderDecl &arg) {
              TRY_TO(traverseNode(arg));
              return true;
            },
            [&](const AggregateDestructuration &arg) {
              TRY_TO(traverseNode(arg));
              return true;
            },
            [&](const UnionDestructuration &arg) {
              TRY_TO(traverseNode(arg));
              return true;
            },
            [](const auto &arg) -> std::unique_ptr<ExprMatchCaseLhsVal> {
              llvm_unreachable("All cases covered");
            }},
        lhsVals);
  }

  bool traverseExprMatchCase(const ExprMatchCase &matchCase) {
    TRY_TO(visitNode(matchCase));

    auto lhsVisitors =
        source::visitors{[&](const Otherwise &arg) { return true; },
                         [&](const ExprMatchCaseLhsVal &arg) {
                           return TraverseExprMatchCaseLhsVal(arg);
                         }};

    if (!std::visit(lhsVisitors, matchCase.getLhs())) {
      return false;
    }

    TRY_TO(traverseNode(matchCase.getRhs()));

    TRY_TO(walkupNode(matchCase));
    return true;
  }

  bool traverseAggregateDestructuration(
      const AggregateDestructuration &aggregateDes) {
    TRY_TO(visitNode(aggregateDes));
    for (auto &elem : aggregateDes.getElems()) {
      TRY_TO(traverseNode(elem));
    }
    TRY_TO(walkupNode(aggregateDes));
    return true;
  }

  bool traverseAggregateDestructurationElem(
      const AggregateDestructurationElem &aggregateDesElem) {
    TRY_TO(visitNode(aggregateDesElem));
    TRY_TO(walkupNode(aggregateDesElem));
    return TraverseExprMatchCaseLhsVal(aggregateDesElem.getValue());
  }

  bool traverseUnionAlternativeFieldDecl(
      const UnionAlternativeFieldDecl &alternativeFieldDecl) {
    TRY_TO(visitNode(alternativeFieldDecl));
    return true;
  }

  bool traverseUnionAlternativeDecl(const UnionAlternativeDecl &alternative) {
    TRY_TO(visitNode(alternative));
    for (const auto &field : alternative.getFields()) {
      TRY_TO(traverseNode(field));
    }
    return true;
  }

  bool traverseUnionDecl(const UnionDecl &enumDecl) {
    TRY_TO(visitNode(enumDecl));
    for (const auto &alternative : enumDecl.getAlternatives()) {
      TRY_TO(traverseNode(alternative));
    }
    return true;
  }

  bool traverseUnionDestructuration(const UnionDestructuration &unionDes) {
    TRY_TO(visitNode(unionDes));
    TRY_TO(traverseNode(unionDes.getDestructuredData()));
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
