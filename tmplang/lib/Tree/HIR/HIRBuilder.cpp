#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Casting.h>
#include <tmplang/Tree/Source/CompilationUnit.h>

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
  Optional<FunctionDecl> get(const source::FunctionDecl &);
  Optional<ParamDecl> get(const source::ParamDecl &);

  const Type *get(const source::Type &);

private:
  HIRContext &Ctx;
  SymbolTable SymTable;
};

const hir::Type *HIRBuilder::get(const source::Type &type) {
  switch (type.getKind()) {
  case source::Type::NamedType:
    return BuiltinType::get(Ctx, cast<source::NamedType>(&type)->getName());
  case source::Type::TupleType:
    const auto &tupleType = *cast<source::TupleType>(&type);
    SmallVector<const hir::Type *> types;
    llvm::transform(tupleType.getTypes(), std::back_inserter(types),
                    [&](const source::RAIIType &type) { return get(*type); });

    if (llvm::any_of(types, [](const Type *type) { return type == nullptr; })) {
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

static FunctionDecl::FunctionKind GetFunctionKind(Token tk) {
  assert(tk.isOneOf(TK_ProcType, TK_FnType));
  if (tk.is(TK_ProcType)) {
    return FunctionDecl::proc;
  }
  return FunctionDecl::fn;
}

Optional<FunctionDecl> HIRBuilder::get(const source::FunctionDecl &srcFunc) {
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

  // TODO: Process body

  return FunctionDecl(srcFunc, srcFunc.getName(),
                      GetFunctionKind(srcFunc.getFuncType()), *hirReturnType,
                      std::move(paramList));
}

Optional<CompilationUnit>
HIRBuilder::build(const source::CompilationUnit &compUnit) {
  CompilationUnit result(compUnit);

  // Push the global scope
  SymTable.pushScope();

  for (const source::FunctionDecl &srcFunc : compUnit.getFunctionDecls()) {
    auto hirFunc = get(srcFunc);
    if (!hirFunc) {
      return None;
    }

    result.addFunctionDecl(std::move(*hirFunc));
  }

  return result;
}

} // namespace

Optional<CompilationUnit>
tmplang::hir::buildHIR(const source::CompilationUnit &compUnit,
                       HIRContext &ctx) {
  return HIRBuilder(ctx).build(compUnit);
}
