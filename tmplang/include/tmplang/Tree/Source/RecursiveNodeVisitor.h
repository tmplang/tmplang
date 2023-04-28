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
  bool traversePlaceholderDecl(const PlaceholderDecl &placeholderDecl) {
    TRY_TO(visitNode(placeholderDecl));
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
  bool traverseExprAggregateDataAccess(
      const ExprAggregateDataAccess &exprDataField) {
    TRY_TO(visitNode(exprDataField));
    TRY_TO(traverseNode(exprDataField.getBase()));
    return true;
  }
  bool traverseExprMatch(const ExprMatch &exprMatch) {
    TRY_TO(visitNode(exprMatch));
    TRY_TO(traverseNode(exprMatch.getMatchedExpr()));
    for (const ExprMatchCase &matchCase : exprMatch.getCases()) {
      TRY_TO(traverseNode(matchCase));
    }
    return true;
  }

  bool traverseExprMatchCase(const ExprMatchCase &matchCase) {
    TRY_TO(visitNode(matchCase));

    if (auto *otherwise = std::get_if<Otherwise>(&matchCase.getLhs())) {
      TRY_TO(traverseNode(*otherwise));
    } else {
      auto visitors = source::visitors{[&](const std::unique_ptr<Expr> &val) {
                                         TRY_TO(traverseNode(*val));
                                         return true;
                                       },
                                       [&](const auto &val) {
                                         TRY_TO(traverseNode(val));
                                         return true;
                                       }};
      const auto &lhsVal =
          std::get<std::unique_ptr<ExprMatchCaseLhsVal>>(matchCase.getLhs());
      if (!std::visit(visitors, *lhsVal)) {
        return false;
      }
    }

    TRY_TO(traverseNode(*matchCase.getRhs()));

    return true;
  }

  bool traverseVoidPlaceholder(const VoidPlaceholder &placeholder) {
    TRY_TO(visitNode(placeholder));
    return true;
  }

  bool traverseOtherwise(const Otherwise &otherwise) {
    TRY_TO(visitNode(otherwise));
    return true;
  }

  bool traverseDataDestructuration(const DataDestructuration &dataDes) {
    TRY_TO(visitNode(dataDes));

    for (const auto &dataDesElem : dataDes.DataElems) {
      TRY_TO(traverseNode(dataDesElem));
    }

    return true;
  }

  bool traverseTupleDestructuration(const TupleDestructuration &tupleDes) {
    TRY_TO(visitNode(tupleDes));

    for (const auto &tupleDesElem : tupleDes.getTupleElems()) {
      TRY_TO(traverseNode(tupleDesElem));
    }

    return true;
  }

  bool
  traverseDataDestructurationElem(const DataDestructurationElem &dataDesElem) {
    TRY_TO(visitNode(dataDesElem));

    auto visitors = source::visitors{[&](const std::unique_ptr<Expr> &val) {
                                       TRY_TO(traverseNode(*val));
                                       return true;
                                     },
                                     [&](const auto &val) {
                                       TRY_TO(traverseNode(val));
                                       return true;
                                     }};
    if (!std::visit(visitors, dataDesElem.getValue())) {
      return false;
    }

    return true;
  }

  bool traverseTupleDestructurationElem(
      const TupleDestructurationElem &tupleDesElem) {
    TRY_TO(visitNode(tupleDesElem));

    auto visitors = source::visitors{[&](const std::unique_ptr<Expr> &val) {
                                       TRY_TO(traverseNode(*val));
                                       return true;
                                     },
                                     [&](const auto &val) {
                                       TRY_TO(traverseNode(val));
                                       return true;
                                     }};

    if (!std::visit(visitors, tupleDesElem.getValue())) {
      return false;
    }

    return true;
  }

  bool traverseUnionDecl(const UnionDecl &unionDecl) {
    TRY_TO(visitNode(unionDecl));

    for (const auto &alternative : unionDecl.getAlternatives()) {
      TRY_TO(traverseNode(alternative));
    }

    return true;
  }

  bool traverseUnionAlternativeDecl(const UnionAlternativeDecl &alternative) {
    TRY_TO(visitNode(alternative));

    for (const auto &field : alternative.getFields()) {
      TRY_TO(traverseNode(field));
    }

    return true;
  }

  bool traverseUnionAlternativeFieldDecl(
      const UnionAlternativeFieldDecl &alternativeField) {
    TRY_TO(visitNode(alternativeField));
    return true;
  }

  bool traverseUnionDestructuration(const UnionDestructuration &unionDes) {
    TRY_TO(visitNode(unionDes));
    TRY_TO(traverseNode(unionDes.getDataDestructuration()));
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
