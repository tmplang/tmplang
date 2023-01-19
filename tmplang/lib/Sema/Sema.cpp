#include <tmplang/Sema/Sema.h>

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
/// Make sure all returned values type match the fuction result type where
/// they dwell in.
//=-------------------------------------------------------------------------=//
class AssertReturnMatchesTypeOfSubprogram
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<AssertReturnMatchesTypeOfSubprogram> {
public:
  using SemaAnalysisPass::SemaAnalysisPass;

  bool visitSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    CurrentFuncRetTy = &subprogramDecl.getSubprogramType().getReturnType();
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
//=                       AssertReturnsInAllCFPaths                         =//
//=-------------------------------------------------------------------------=//
/// Make sure that for a subprogram that its return type is diferent from unit
/// there is a ret expression in all possible paths. So no path is without ret
//=-------------------------------------------------------------------------=//
class AssertReturnsInAllCFPaths
    : public SemaAnalysisPass,
      public RecursiveASTVisitor<AssertReturnsInAllCFPaths> {
public:
  using SemaAnalysisPass::SemaAnalysisPass;

  bool traverseSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    // If the result type is unit type, there is no need to return in all paths
    auto *retTy = dyn_cast<hir::TupleType>(
        &subprogramDecl.getSubprogramType().getReturnType());
    if (retTy && retTy->isUnit()) {
      return true;
    }

    const bool returnsInAllPaths =
        any_of(subprogramDecl.getBody(), [](const std::unique_ptr<Expr> &expr) {
          // Right now, since there are no if-else or any other kind of
          // expresion that may modify the control flow, this is enough
          return expr->getKind() == Node::Kind::ExprRet;
        });

    if (!returnsInAllPaths) {
      Diagnostic(DiagId::err_subprogram_does_not_return_in_all_paths,
                 {subprogramDecl.getBeginLoc(), subprogramDecl.getEndLoc()},
                 NoHint())
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
  RunPass(AssertReturnMatchesTypeOfSubprogram);

  return !didAnyPassEmitAnError;
}
