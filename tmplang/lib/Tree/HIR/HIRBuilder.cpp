#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Casting.h>
#include <tmplang/Tree/HIR/Exprs.h>
#include <tmplang/Tree/Source/CompilationUnit.h>
#include <tmplang/Tree/Source/Exprs.h>

using namespace tmplang;
using namespace tmplang::hir;

namespace {

class SymbolTable {
  struct Scope {
    llvm::DenseSet<StringRef> Identifiers;
  };

public:
  SymbolTable() {}

  bool insertIdentifier(StringRef id) {
    // FIXME: This is very restrictive, this is not discriminating between
    // functions, variables, macros, types, ...; just identifiers :facepalm:
    //
    // Also, in case we want to support more advance shadowings systems like
    // a name binding approach (like Rust does), this is completly wrong.
    //
    // When a decisison is taken, adapt the symbol table to reflect that. For
    // now, just check that no param repeats within the same function, and that
    // no function repeats the same name
    return ScopeStack.back().Identifiers.insert(id).second;
  }

  void pushScope() { ScopeStack.emplace_back(); }
  void popScope() { ScopeStack.pop_back(); }

private:
  SmallVector<Scope> ScopeStack;
};

class HIRBuilder {
public:
  HIRBuilder(HIRContext &ctx) : Ctx(ctx) {}

  Optional<CompilationUnit> build(const source::CompilationUnit &);

private:
  Optional<SubprogramDecl> get(const source::SubprogramDecl &);
  Optional<ParamDecl> get(const source::ParamDecl &);

  std::unique_ptr<Expr> get(const source::Expr &);
  std::unique_ptr<ExprIntegerNumber> get(const source::ExprIntegerNumber &);
  std::unique_ptr<ExprRet> get(const source::ExprRet &);
  std::unique_ptr<ExprTuple> get(const source::ExprTuple &);

  const Type *get(const source::Type &);

private:
  HIRContext &Ctx;
  SymbolTable SymTable;
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
  if (!SymTable.insertIdentifier(srcParamDecl.getName())) {
    return None;
  }

  const Type *type = get(srcParamDecl.getType());
  if (!type) {
    return None;
  }

  return ParamDecl(srcParamDecl, srcParamDecl.getName(), *type);
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
  if (!SymTable.insertIdentifier(srcFunc.getName())) {
    return None;
  }

  std::vector<ParamDecl> paramList;

  SymTable.pushScope();
  for (const source::ParamDecl &param : srcFunc.getParams()) {
    auto hirParamDecl = get(param);
    if (!hirParamDecl) {
      return None;
    }
    paramList.push_back(std::move(*hirParamDecl));
  }
  SymTable.popScope();

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

  std::vector<std::unique_ptr<Expr>> bodyExprs;

  SymTable.pushScope();
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
  SymTable.popScope();

  SmallVector<const Type *> paramTys;
  transform(paramList, std::back_inserter(paramTys),
            [](const ParamDecl &paramDecl) { return &paramDecl.getType(); });

  return SubprogramDecl(srcFunc, srcFunc.getName(),
                        GetFunctionKind(srcFunc.getFuncType()),
                        SubprogramType::get(Ctx, *hirReturnType, paramTys),
                        std::move(paramList), std::move(bodyExprs));
}

Optional<CompilationUnit>
HIRBuilder::build(const source::CompilationUnit &compUnit) {
  CompilationUnit result(compUnit);

  // Push the global scope
  SymTable.pushScope();

  for (const source::SubprogramDecl &srcFunc : compUnit.getSubprogramDecls()) {
    auto hirFunc = get(srcFunc);
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
