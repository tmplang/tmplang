#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <llvm/Support/Casting.h>
#include <tmplang/Tree/HIR/Exprs.h>
#include <tmplang/Tree/Source/CompilationUnit.h>
#include <tmplang/Tree/Source/Exprs.h>

using namespace tmplang;
using namespace tmplang::hir;

namespace {

class HIRBuilder {
public:
  HIRBuilder(HIRContext &ctx)
      : Ctx(ctx), SymMan(ctx.getSymbolManager()),
        ScopeStack(1, &SymMan.getGlobalScope()) {}

  Optional<CompilationUnit> build(const source::CompilationUnit &);

private:
  Optional<SubprogramDecl> get(const source::SubprogramDecl &);
  Optional<ParamDecl> get(const source::ParamDecl &);

  std::unique_ptr<Expr> get(const source::Expr &);
  std::unique_ptr<ExprIntegerNumber> get(const source::ExprIntegerNumber &);
  std::unique_ptr<ExprRet> get(const source::ExprRet &);
  std::unique_ptr<ExprTuple> get(const source::ExprTuple &);

  const Type *get(const source::Type &);

  Symbol &addSymbolToCurrentScope(const SymbolKind kind, llvm::StringRef id,
                                  const Type &ty) {
    auto &sym = SymMan.createSymbol(kind, id, ty);
    ScopeStack.back()->addSymbol(sym);
    return sym;
  }

  SymbolicScope &pushSymbolicScope() {
    SymbolicScope &newScope = SymMan.createSymbolicScope(*ScopeStack.back());
    return *ScopeStack.emplace_back(&newScope);
  }

  void popSymbolScope() { ScopeStack.pop_back(); }

  SymbolicScope &currentSymScope() { return *ScopeStack.back(); }

private:
  HIRContext &Ctx;
  SymbolManager &SymMan;
  /// Stack of scopes we are currently in. SymbolicScopes live in the
  /// SymbolicManager. TODO: Profile to know what size to give initialy
  SmallVector<SymbolicScope *, 6> ScopeStack;
};

std::unique_ptr<ExprIntegerNumber>
HIRBuilder::get(const source::ExprIntegerNumber &exprIntNum) {
  int32_t num = 0;
  for (const char c : exprIntNum.getNumber().getLexeme()) {
    if (c == '_') {
      continue;
    }
    assert(llvm::is_contained("0123456789", c) && "Expected a number");

    num *= 10;
    num += c - '0';
  }

  // FIXME: For now all numbers are signed 32 bits
  return std::make_unique<hir::ExprIntegerNumber>(
      exprIntNum, hir::BuiltinType::get(Ctx, BuiltinType::K_i32),
      llvm::APInt(32, num, /*isSigned=*/true));
}

std::unique_ptr<ExprRet> HIRBuilder::get(const source::ExprRet &exprRet) {
  auto returnedExpr =
      exprRet.getReturnedExpr() ? get(*exprRet.getReturnedExpr()) : nullptr;

  return std::make_unique<hir::ExprRet>(
      exprRet, returnedExpr ? returnedExpr->getType() : TupleType::getUnit(Ctx),
      std::move(returnedExpr));
}

std::unique_ptr<ExprTuple> HIRBuilder::get(const source::ExprTuple &exprTuple) {
  SmallVector<std::unique_ptr<hir::Expr>, 4> values;
  transform(exprTuple.getVals(), std::back_inserter(values),
            [&](const source::TupleElem &elem) { return get(elem.getVal()); });

  SmallVector<const Type *> tupleTys;
  tupleTys.reserve(values.size());
  transform(values, std::back_inserter(tupleTys),
            [&](std::unique_ptr<hir::Expr> &expr) { return &expr->getType(); });

  return std::make_unique<hir::ExprTuple>(
      exprTuple, TupleType::get(Ctx, tupleTys), std::move(values));
}

std::unique_ptr<Expr> HIRBuilder::get(const source::Expr &expr) {
  switch (expr.getKind()) {
  case source::Node::Kind::ExprTuple:
    return get(*cast<source::ExprTuple>(&expr));
  case source::Node::Kind::ExprIntegerNumber:
    return get(*cast<source::ExprIntegerNumber>(&expr));
  case source::Node::Kind::ExprRet:
    return get(*cast<source::ExprRet>(&expr));
  default:
    break;
  }
  llvm_unreachable("This should not be reachable");
}

const hir::Type *HIRBuilder::get(const source::Type &type) {
  switch (type.getKind()) {
  case source::Type::NamedType:
    return BuiltinType::get(Ctx, cast<source::NamedType>(&type)->getName());
  case source::Type::TupleType:
    const auto &tupleType = *cast<source::TupleType>(&type);
    SmallVector<const hir::Type *> types;
    transform(tupleType.getTypes(), std::back_inserter(types),
              [&](const source::RAIIType &type) { return get(*type); });

    if (any_of(types, [](const Type *type) { return type == nullptr; })) {
      return nullptr;
    }
    return &TupleType::get(Ctx, types);
  }
  llvm_unreachable("All cases covered");
}

Optional<ParamDecl> HIRBuilder::get(const source::ParamDecl &srcParamDecl) {
  const Type *type = get(srcParamDecl.getType());
  if (!type) {
    return None;
  }

  if (currentSymScope().containsSymbol(SymbolKind::ParamOrVarDecl,
                                       srcParamDecl.getName())) {
    // FIXME: Report error
    return None;
  }
  Symbol &sym = addSymbolToCurrentScope(SymbolKind::ParamOrVarDecl,
                                        srcParamDecl.getName(), *type);

  return ParamDecl(srcParamDecl, sym);
}

static SubprogramDecl::FunctionKind GetFunctionKind(Token tk) {
  assert(tk.isOneOf(TK_ProcType, TK_FnType));
  if (tk.is(TK_ProcType)) {
    return SubprogramDecl::proc;
  }
  return SubprogramDecl::fn;
}

Optional<SubprogramDecl>
HIRBuilder::get(const source::SubprogramDecl &srcFunc) {
  const Type *hirReturnType = nullptr;
  if (const source::Type *srcType = srcFunc.getReturnType()) {
    // FIXME: Right now we only have builtin types, so this is complete. Once
    // we start adding user defined types or other more complex types, change
    // this to reflect that
    hirReturnType = get(*srcType);
  } else {
    hirReturnType = &TupleType::getUnit(Ctx);
  }
  if (!hirReturnType) {
    return None;
  }

  // Add new scope for params and body
  pushSymbolicScope();

  std::vector<ParamDecl> paramList;
  for (const source::ParamDecl &param : srcFunc.getParams()) {
    auto hirParamDecl = get(param);
    if (!hirParamDecl) {
      return None;
    }
    paramList.push_back(std::move(*hirParamDecl));
  }

  std::vector<std::unique_ptr<Expr>> bodyExprs;
  for (const source::ExprStmt &expr : srcFunc.getBlock().Exprs) {
    auto *srcExpr = expr.getExpr();
    if (!srcExpr) {
      continue;
    }

    std::unique_ptr<hir::Expr> hirExpr = get(*srcExpr);
    if (!hirExpr) {
      return None;
    }

    bodyExprs.push_back(std::move(hirExpr));
  }

  // Pop params and body scope, so we are back previous scope for next
  // declarations
  popSymbolScope();

  SmallVector<const Type *> paramTys;
  transform(paramList, std::back_inserter(paramTys),
            [](const ParamDecl &paramDecl) { return &paramDecl.getType(); });
  if (currentSymScope().containsSymbol(SymbolKind::Subprogram,
                                       srcFunc.getName())) {
    // FIXME: Report error
    return None;
  }
  Symbol &funcSym = addSymbolToCurrentScope(
      SymbolKind::Subprogram, srcFunc.getName(),
      SubprogramType::get(Ctx, *hirReturnType, paramTys));

  return SubprogramDecl(srcFunc, funcSym,
                        GetFunctionKind(srcFunc.getFuncType()),
                        std::move(paramList), std::move(bodyExprs));
}

Optional<CompilationUnit>
HIRBuilder::build(const source::CompilationUnit &compUnit) {
  CompilationUnit result(compUnit);

  for (const std::unique_ptr<tmplang::source::Decl> &srcFunc :
       compUnit.getTopLevelDecls()) {
    auto *subprog = dyn_cast<source::SubprogramDecl>(srcFunc.get());
    if (!subprog) {
      // TODO: Build more things than just SubprogramDecl
      continue;
    }

    auto hirFunc = get(*subprog);
    if (!hirFunc) {
      return None;
    }

    result.addSubprogramDecl(std::move(*hirFunc));
  }

  return result;
}

} // namespace

Optional<CompilationUnit>
tmplang::hir::buildHIR(const source::CompilationUnit &compUnit,
                       HIRContext &ctx) {
  return HIRBuilder(ctx).build(compUnit);
}
