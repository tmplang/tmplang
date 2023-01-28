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
  std::unique_ptr<Decl> getTopLevelDecl(const source::Decl &);

  std::unique_ptr<Decl> get(const source::SubprogramDecl &);
  Optional<ParamDecl> get(const source::ParamDecl &);
  std::unique_ptr<Decl> get(const source::DataDecl &);
  Optional<DataFieldDecl> get(const source::DataFieldDecl &);

  std::unique_ptr<Expr> get(const source::Expr &);
  std::unique_ptr<ExprIntegerNumber> get(const source::ExprIntegerNumber &);
  std::unique_ptr<ExprRet> get(const source::ExprRet &);
  std::unique_ptr<ExprTuple> get(const source::ExprTuple &);
  std::unique_ptr<ExprVarRef> get(const source::ExprVarRef &);

  const Type *get(const source::Type &);

  Symbol *fetchSymbolRecursively(SymbolKind kind, llvm::StringRef id) const;
  bool isSymbolInCurrentScope(SymbolKind kind, llvm::StringRef id) const;
  Symbol &addSymbolToCurrentScope(SymbolKind kind, llvm::StringRef id,
                                  const Type &ty);

  SymbolicScope &pushSymbolicScope() {
    SymbolicScope &newScope = SymMan.createSymbolicScope();
    return *ScopeStack.emplace_back(&newScope);
  }

  void popSymbolScope() { ScopeStack.pop_back(); }

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

std::unique_ptr<ExprVarRef>
HIRBuilder::get(const source::ExprVarRef &exprVarRef) {
  auto *sym = fetchSymbolRecursively(SymbolKind::ReferenciableFromExprVarRef,
                                     exprVarRef.getName());
  if (!sym) {
    // TODO: Emit error about var not defined
    return nullptr;
  }

  return std::make_unique<ExprVarRef>(exprVarRef, *sym);
}

std::unique_ptr<Expr> HIRBuilder::get(const source::Expr &expr) {
  switch (expr.getKind()) {
  case source::Node::Kind::ExprTuple:
    return get(*cast<source::ExprTuple>(&expr));
  case source::Node::Kind::ExprIntegerNumber:
    return get(*cast<source::ExprIntegerNumber>(&expr));
  case source::Node::Kind::ExprRet:
    return get(*cast<source::ExprRet>(&expr));
  case source::Node::Kind::ExprVarRef:
    return get(*cast<source::ExprVarRef>(&expr));
  default:
    break;
  }
  llvm_unreachable("This should not be reachable");
}

const hir::Type *HIRBuilder::get(const source::Type &type) {
  switch (type.getKind()) {
  case source::Type::NamedType: {
    auto *sym =
        fetchSymbolRecursively(SymbolKind::ReferenciableFromType,
                               llvm::cast<source::NamedType>(&type)->getName());
    if (!sym) {
      // TODO: Emit error (type undefined)
      return nullptr;
    }

    return &sym->getType();
  }
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

Symbol *HIRBuilder::fetchSymbolRecursively(SymbolKind kind,
                                           llvm::StringRef id) const {
  for (SymbolicScope *symScope : llvm::reverse(ScopeStack)) {
    auto *sym = llvm::find_if(symScope->getSymbols(), [=](const Symbol &sym) {
      return sym.getKind() == kind && sym.getId() == id;
    });

    if (sym != symScope->getSymbols().end()) {
      return &sym->get();
    }
  }

  return nullptr;
}

bool HIRBuilder::isSymbolInCurrentScope(SymbolKind kind,
                                        llvm::StringRef id) const {
  return ScopeStack.back()->containsSymbol(kind, id);
}

Symbol &HIRBuilder::addSymbolToCurrentScope(SymbolKind kind, llvm::StringRef id,
                                            const Type &ty) {
  assert(!isSymbolInCurrentScope(kind, id));

  auto &sym = SymMan.createSymbol(kind, id, ty);
  ScopeStack.back()->addSymbol(sym);
  return sym;
}

Optional<ParamDecl> HIRBuilder::get(const source::ParamDecl &srcParamDecl) {
  const Type *type = get(srcParamDecl.getType());
  if (!type) {
    return None;
  }

  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
                             srcParamDecl.getName())) {
    // FIXME: Report error
    return None;
  }
  Symbol &sym = addSymbolToCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
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

std::unique_ptr<Decl> HIRBuilder::getTopLevelDecl(const source::Decl &decl) {
  switch (decl.getKind()) {
  case source::Node::Kind::SubprogramDecl:
    return get(*cast<source::SubprogramDecl>(&decl));
  case source::Node::Kind::DataDecl:
    return get(*cast<source::DataDecl>(&decl));
  case source::Node::Kind::CompilationUnit:
  case source::Node::Kind::DataFieldDecl:
  case source::Node::Kind::ParamDecl:
  case source::Node::Kind::ExprStmt:
  case source::Node::Kind::ExprIntegerNumber:
  case source::Node::Kind::ExprRet:
  case source::Node::Kind::ExprDataFieldAccess:
  case source::Node::Kind::ExprTuple:
  case source::Node::Kind::ExprVarRef:
  case source::Node::Kind::TupleElem:
    // All these nodes cannot be top level decls
    break;
  }
  llvm_unreachable("These case are not possible");
}

std::unique_ptr<Decl> HIRBuilder::get(const source::SubprogramDecl &srcFunc) {
  const Type *hirReturnType = nullptr;
  if (const source::Type *srcType = srcFunc.getReturnType()) {
    hirReturnType = get(*srcType);
  } else {
    hirReturnType = &TupleType::getUnit(Ctx);
  }
  if (!hirReturnType) {
    return nullptr;
  }

  // Add new scope for params and body
  pushSymbolicScope();

  std::vector<ParamDecl> paramList;
  for (const source::ParamDecl &param : srcFunc.getParams()) {
    auto hirParamDecl = get(param);
    if (!hirParamDecl) {
      return nullptr;
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
      return nullptr;
    }

    bodyExprs.push_back(std::move(hirExpr));
  }

  // Pop params and body scope, so we are back previous scope for next
  // declarations
  popSymbolScope();

  SmallVector<const Type *> paramTys;
  transform(paramList, std::back_inserter(paramTys),
            [](const ParamDecl &paramDecl) { return &paramDecl.getType(); });
  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromExprCall,
                             srcFunc.getName())) {
    // FIXME: Report error
    return nullptr;
  }
  Symbol &funcSym = addSymbolToCurrentScope(
      SymbolKind::ReferenciableFromExprCall, srcFunc.getName(),
      SubprogramType::get(Ctx, *hirReturnType, paramTys));

  return std::make_unique<SubprogramDecl>(
      srcFunc, funcSym, GetFunctionKind(srcFunc.getFuncType()),
      std::move(paramList), std::move(bodyExprs));
}

std::unique_ptr<Decl> HIRBuilder::get(const source::DataDecl &dataDecl) {
  // Add new scope for data fields
  pushSymbolicScope();

  std::vector<DataFieldDecl> fields;
  for (const source::DataFieldDecl &dataFieldDecl : dataDecl.getFields()) {
    auto hirFieldDecl = get(dataFieldDecl);
    if (!hirFieldDecl) {
      return nullptr;
    }
    fields.emplace_back(std::move(*hirFieldDecl));
  }

  SmallVector<const Type *> fieldTys;
  fieldTys.reserve(fields.size());
  transform(fields, std::back_inserter(fieldTys),
            [](const DataFieldDecl &field) { return &field.getType(); });

  popSymbolScope();

  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromType,
                             dataDecl.getName())) {
    // FIXME: Report error
    return nullptr;
  }
  Symbol &dataDeclSym = addSymbolToCurrentScope(
      SymbolKind::ReferenciableFromType, dataDecl.getName(),
      DataType::get(Ctx, dataDecl.getName(), fieldTys));
  return std::make_unique<DataDecl>(dataDecl, dataDeclSym, std::move(fields));
}

Optional<DataFieldDecl>
HIRBuilder::get(const source::DataFieldDecl &dataFieldDecl) {
  auto *ty = get(dataFieldDecl.getType());
  if (!ty) {
    return None;
  }

  // FIXME: In reality these should not be stored since they are no Types
  // and do not provide much value. We are resuing the simbolic scopes
  // to block redefinitions of the same data file decl
  if (isSymbolInCurrentScope(SymbolKind::UnreferenciableDataFieldDecl,
                             dataFieldDecl.getName())) {
    // FIXME: Report error
    return None;
  }
  Symbol &dataFieldSym = addSymbolToCurrentScope(
      SymbolKind::UnreferenciableDataFieldDecl, dataFieldDecl.getName(), *ty);

  return DataFieldDecl(dataFieldDecl, dataFieldSym);
}

static void AddNamedBuiltinTypes(SymbolManager &sm, HIRContext &ctx) {
  SymbolicScope &symScope = sm.getGlobalScope();

  constexpr BuiltinType::Kind builtinTypes[] = {BuiltinType::Kind::K_i32};
  for (BuiltinType::Kind kind : builtinTypes) {
    auto &sym = sm.createSymbol(SymbolKind::ReferenciableFromType,
                                ToString(kind), BuiltinType::get(ctx, kind));
    symScope.addSymbol(sym);
  }
}

Optional<CompilationUnit>
HIRBuilder::build(const source::CompilationUnit &compUnit) {
  CompilationUnit result(compUnit);

  AddNamedBuiltinTypes(SymMan, Ctx);

  for (const std::unique_ptr<tmplang::source::Decl> &srcDecl :
       compUnit.getTopLevelDecls()) {

    auto decl = getTopLevelDecl(*srcDecl);
    if (!decl) {
      return None;
    }

    result.addDecl(std::move(decl));
  }

  return result;
}

} // namespace

Optional<CompilationUnit>
tmplang::hir::buildHIR(const source::CompilationUnit &compUnit,
                       HIRContext &ctx) {
  return HIRBuilder(ctx).build(compUnit);
}
