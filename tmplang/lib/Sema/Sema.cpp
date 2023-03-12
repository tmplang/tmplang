#include <tmplang/Sema/Sema.h>

#include <llvm/ADT/ScopeExit.h>
#include <tmplang/Diagnostics/Diagnostic.h>
#include <tmplang/Diagnostics/Hint.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/CompilationUnit.h>
#include <tmplang/Tree/HIR/RecursiveNodeVisitor.h>
#include <tmplang/Tree/HIR/Types.h>

using namespace tmplang;
using namespace tmplang::hir;

namespace {

/// Base class that all Sema analysis passes should inherit from. Contains the
/// factored logic to mark a pass as failed, and the required fields to report
/// diagnostics.
class SemaAnalysisPass {
public:
  SemaAnalysisPass(const SourceManager &sm, raw_ostream &out)
      : SM(sm), Out(out) {}

  bool didEmitAnError() const { return DidAnErrorOcurred; };
  void markAsFailure() { DidAnErrorOcurred = true; };

  const SourceManager &getSourceManager() const { return SM; }
  raw_ostream &outs() const { return Out; }

private:
  bool DidAnErrorOcurred = false;
  const SourceManager &SM;
  raw_ostream &Out;
};
} // namespace

namespace {

//=-------------------------------------------------------------------------=//
//=                      Begin Semantic Analysis Passes                     =//
//=-------------------------------------------------------------------------=//

//=-------------------------------------------------------------------------=//
//=                 AssertReturnMatchesTypeOfSubprogram                     =//
//=-------------------------------------------------------------------------=//
// Make sure all returned values type match the fuction result type where
// they dwell in.
//=-------------------------------------------------------------------------=//
class AssertReturnMatchesTypeOfSubprogram
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<AssertReturnMatchesTypeOfSubprogram> {
public:
  using SemaAnalysisPass::SemaAnalysisPass;

  bool visitSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    CurrentFuncRetTy = &subprogramDecl.getType().getReturnType();
    return true;
  }

  bool traverseExprRet(const ExprRet &retExpr) {
    if (&retExpr.getType() != CurrentFuncRetTy) {
      Diagnostic(DiagId::err_mistmach_between_ret_and_subprogram_type,
                 {retExpr.getBeginLoc(), retExpr.getEndLoc()}, NoHint())
          .print(outs(), getSourceManager());
      markAsFailure();
    }
    return true;
  }

private:
  const Type *CurrentFuncRetTy = nullptr;
};

//=-------------------------------------------------------------------------=//
//=                     AssertMarchCasesDoesNotReturn                       =//
//=-------------------------------------------------------------------------=//
// Make sure that no case of any match uses the ret expr
//=-------------------------------------------------------------------------=//

class AssertMarchCasesDoesNotReturn
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<AssertMarchCasesDoesNotReturn> {
public:
  using SemaAnalysisPass::SemaAnalysisPass;

  bool visitExprMatchCase(const ExprMatchCase &matchCase) {
    if (isa<ExprRet>(matchCase.getRhs())) {
      // TODO: Add diagnostics message
      markAsFailure();
    }
    return true;
  }
};

//=-------------------------------------------------------------------------=//
//=                       AssertReturnsInAllCFPaths                         =//
//=-------------------------------------------------------------------------=//
// Make sure that for a subprogram that its return type is diferent from unit
// there is a ret expression in all possible paths. So no path is without ret
//=-------------------------------------------------------------------------=//
class AssertReturnsInAllCFPaths
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<AssertReturnsInAllCFPaths> {
public:
  using Base = RecursiveASTVisitor<AssertReturnsInAllCFPaths>;

  struct ControlFlowBifurcation {
    ControlFlowBifurcation(ControlFlowBifurcation *p = nullptr) : Parent(p) {}

    bool DoesReturnsInAllPath() const {
      return PathExitsFunction &&
             all_of(Bifurcations,
                    [](const std::unique_ptr<ControlFlowBifurcation> &cfp) {
                      return cfp->DoesReturnsInAllPath();
                    });
    }

    bool PathExitsFunction = false;
    ControlFlowBifurcation *Parent = nullptr;
    SmallVector<std::unique_ptr<ControlFlowBifurcation>> Bifurcations;
  };

  AssertReturnsInAllCFPaths(const SourceManager &sm, raw_ostream &out)
      : SemaAnalysisPass(sm, out), CFP(), CurrentCFP(&CFP) {}

  bool walkupSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    auto resetOnExit = llvm::make_scope_exit([&]() {
      // Reset control flow bifurcations
      CFP = ControlFlowBifurcation();
      CurrentCFP = &CFP;
    });

    auto *retTy =
        dyn_cast<hir::TupleType>(&subprogramDecl.getType().getReturnType());
    if (retTy && retTy->isUnit()) {
      return true;
    }

    if (!CFP.DoesReturnsInAllPath()) {
      Diagnostic(DiagId::err_subprogram_does_not_return_in_all_paths,
                 {subprogramDecl.getBeginLoc(), subprogramDecl.getEndLoc()},
                 NoHint())
          .print(outs(), getSourceManager());
      markAsFailure();
    }

    return true;
  }

  // Use traverse since we don't care about if the expr of the return
  // open a control flow.
  bool traverseExprRet(const ExprRet &ret) {
    CurrentCFP->PathExitsFunction = true;
    return true;
  }

  static bool CanBifurcateControlFlow(Node::Kind k) {
    switch (k) {
    case Node::Kind::ExprMatchCase:
      return true;
    case Node::Kind::CompilationUnit:
    case Node::Kind::SubprogramDecl:
    case Node::Kind::DataFieldDecl:
    case Node::Kind::DataDecl:
    case Node::Kind::ParamDecl:
    case Node::Kind::ExprIntegerNumber:
    case Node::Kind::ExprRet:
    case Node::Kind::ExprVarRef:
    case Node::Kind::ExprAggregateDataAccess:
    case Node::Kind::ExprTuple:
    case Node::Kind::ExprMatch:
    case Node::Kind::PlaceholderDecl:
    case Node::Kind::AggregateDestructuration:
    case Node::Kind::AggregateDestructurationElem:
      return false;
    }
    llvm_unreachable("All cases covered");
  }

  bool visitNode(const Node &node) {
    if (CanBifurcateControlFlow(node.getKind())) {
      CurrentCFP->Bifurcations.push_back(
          std::make_unique<ControlFlowBifurcation>(CurrentCFP));
      CurrentCFP = CurrentCFP->Bifurcations.back().get();
    }

    return Base::visitNode(node);
  }

  bool walkupNode(const Node &node) {
    if (CanBifurcateControlFlow(node.getKind())) {
      CurrentCFP = CurrentCFP->Parent;
    }

    return Base::walkupNode(node);
  }

  ControlFlowBifurcation CFP;
  ControlFlowBifurcation *CurrentCFP;
};

//=-------------------------------------------------------------------------=//
//=                       AllBranchesOfMatchExprAreSameType                 =//
//=-------------------------------------------------------------------------=//
// Make sure that all branches of a match expression return same type
//=-------------------------------------------------------------------------=//
class AllBranchesOfMatchExprAreSameType
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<AllBranchesOfMatchExprAreSameType> {
public:
  using SemaAnalysisPass::SemaAnalysisPass;

  bool visitExprMatch(const ExprMatch &match) {
    ArrayRef<std::unique_ptr<ExprMatchCase>> cases = match.getExprMatchCases();
    assert(!cases.empty());

    const auto &firstBranchTy = cases.front()->getType();
    for (const std::unique_ptr<tmplang::hir::ExprMatchCase> &branch : cases) {
      if (&branch->getRhs().getType() != &firstBranchTy) {
        Diagnostic(
            DiagId::err_match_branch_type_does_not_match_first_branch_type,
            {branch->getRhs().getBeginLoc(), branch->getRhs().getEndLoc()},
            NoHint())
            .print(outs(), getSourceManager());
        markAsFailure();
      }
    }

    return true;
  }
};

//=-------------------------------------------------------------------------=//
//=                       OnlyIntergerExprOnAggregateDestructuration        =//
//=-------------------------------------------------------------------------=//
// Only integers are supported on any placeholder of the lhs of a branch of
// a match expression
//=-------------------------------------------------------------------------=//
class OnlyIntergerExprOnAggregateDestructuration
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<OnlyIntergerExprOnAggregateDestructuration> {
public:
  using SemaAnalysisPass::SemaAnalysisPass;

  bool
  visitAggregateDestructuration(const AggregateDestructuration &destructuing) {
    for (const AggregateDestructurationElem &elem : destructuing.getElems()) {
      auto *expr = std::get_if<std::unique_ptr<Expr>>(&elem.getValue());
      if (!expr) {
        continue;
      }

      if (expr->get()->getKind() != Node::Kind::ExprIntegerNumber) {
        Diagnostic(DiagId::err_match_lhs_branch_is_not_integer,
                   {expr->get()->getBeginLoc(), expr->get()->getEndLoc()},
                   NoHint())
            .print(outs(), getSourceManager());
        markAsFailure();
      }
    }

    return true;
  }
};

//=-------------------------------------------------------------------------=//
//=                       OtherwisePositionalCheck                          =//
//=-------------------------------------------------------------------------=//
// Make sure any match expr contains as last branch the unique possible
// otherwise in the match expression
//=-------------------------------------------------------------------------=//
class OtherwisePositionalCheck
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<OtherwisePositionalCheck> {
public:
  using SemaAnalysisPass::SemaAnalysisPass;

  bool visitExprMatch(const ExprMatch &match) {
    ArrayRef<std::unique_ptr<ExprMatchCase>> cases = match.getExprMatchCases();
    assert(!cases.empty());

    for (const auto &both : llvm::enumerate(cases)) {
      auto &[idx, branch] = both;

      if (std::holds_alternative<Otherwise>(branch->getLhs()) &&
          idx != cases.size() - 1) {
        Diagnostic(DiagId::err_otherwise_can_only_appear_at_last_case,
                   {branch->getBeginLoc(), branch->getEndLoc()}, NoHint())
            .print(outs(), getSourceManager());

        markAsFailure();
      }
    }

    const ExprMatchCase &lastCase = *cases.back();
    if (!std::holds_alternative<Otherwise>(lastCase.getLhs())) {
      Diagnostic(DiagId::err_match_without_final_otherwise_catch_all,
                 {match.getBeginLoc(), match.getEndLoc()}, NoHint())
          .print(outs(), getSourceManager());

      markAsFailure();
    }

    return true;
  }
};

//=-------------------------------------------------------------------------=//
//=                       End Semantic Analysis Passes                      =//
//=-------------------------------------------------------------------------=//

} // namespace

#define RunPass(Pass)                                                          \
  {                                                                            \
    Pass pass(sm, out);                                                        \
    pass.traverseNode(compUnit);                                               \
    didAnyPassEmitAnError |= pass.didEmitAnError();                            \
  }

bool tmplang::Sema(CompilationUnit &compUnit, const SourceManager &sm,
                   raw_ostream &out) {
  bool didAnyPassEmitAnError = false;

  // FIXME: This traverses the tree one time for each pass. We can optimize it
  //        now by calling conviniently traverse instead of visit. A more
  //        future-proof solution would be implemting a dynamic visitor since
  //        the current one uses CRTP and can't be used along with visitors
  //        that are not default constructible
  RunPass(AssertReturnsInAllCFPaths);
  RunPass(AssertMarchCasesDoesNotReturn);
  RunPass(AssertReturnMatchesTypeOfSubprogram);
  RunPass(AllBranchesOfMatchExprAreSameType);
  RunPass(OnlyIntergerExprOnAggregateDestructuration);
  RunPass(OtherwisePositionalCheck);

  return !didAnyPassEmitAnError;
}
