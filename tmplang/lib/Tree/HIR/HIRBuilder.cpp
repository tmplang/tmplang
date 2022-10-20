#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <llvm/ADT/DenseSet.h>
#include <tmplang/Tree/Source/CompilationUnit.h>

using namespace tmplang;
using namespace tmplang::hir;

namespace {

class SymbolTable {
  struct Scope {
    // TODO: This is very restrictive. If we allow overloading based on
    // different signatures,
    //       fix this. Also, this is not discriminating between functions or
    //       variables, just identifiers
    llvm::DenseSet<llvm::StringRef> Identifiers;
  };

public:
  SymbolTable() {}

  bool insertIdentifier(llvm::StringRef id) {
    return ScopeStack.back().Identifiers.insert(id).second;
  }

  void pushScope() { ScopeStack.emplace_back(); }
  void popScope() { ScopeStack.pop_back(); }

private:
  llvm::SmallVector<Scope> ScopeStack;
};

class HIRBuilder {
public:
  HIRBuilder(HIRContext &ctx) : Ctx(ctx) {}

  llvm::Optional<CompilationUnit> build(const source::CompilationUnit &);

private:
  llvm::Optional<FunctionDecl> get(const source::FunctionDecl &);
  llvm::Optional<ParamDecl> get(const source::ParamDecl &);

private:
  HIRContext &Ctx;
  SymbolTable SymTable;
  CompilationUnit Result;
};

llvm::Optional<ParamDecl>
HIRBuilder::get(const source::ParamDecl &srcParamDecl) {
  if (!SymTable.insertIdentifier(srcParamDecl.getName())) {
    return llvm::None;
  }

  const BuiltinType *type =
      BuiltinType::get(Ctx, srcParamDecl.getType().getName());
  if (!type) {
    return llvm::None;
  }

  return ParamDecl(srcParamDecl.getName(), *type);
}

llvm::Optional<FunctionDecl>
HIRBuilder::get(const source::FunctionDecl &srcFunc) {
  if (!SymTable.insertIdentifier(srcFunc.getName())) {
    return llvm::None;
  }

  std::vector<ParamDecl> paramList;

  SymTable.pushScope();
  for (const source::ParamDecl &param : srcFunc.getParams().ParamList) {
    auto hirParamDecl = get(param);
    if (!hirParamDecl) {
      return llvm::None;
    }
    paramList.push_back(std::move(*hirParamDecl));
  }
  SymTable.popScope();

  const Type *hirReturnType = nullptr;
  if (const source::NamedType *srcType = srcFunc.getReturnType()) {
    hirReturnType = BuiltinType::get(Ctx, srcType->getName());
  } else {
    hirReturnType = &BuiltinType::get(Ctx, BuiltinType::K_Unit);
  }

  if (!hirReturnType) {
    return llvm::None;
  }

  // TODO: Process body

  return FunctionDecl(srcFunc.getName(), *hirReturnType, std::move(paramList));
}

llvm::Optional<CompilationUnit>
HIRBuilder::build(const source::CompilationUnit &compUnit) {
  // Push the global scope
  SymTable.pushScope();

  for (const source::FunctionDecl &srcFunc : compUnit.getFunctionDecls()) {
    auto hirFunc = get(srcFunc);
    if (!hirFunc) {
      return llvm::None;
    }

    Result.addFunctionDecl(std::move(*hirFunc));
  }

  return Result;
}

} // namespace

llvm::Optional<CompilationUnit>
tmplang::hir::buildHIR(const source::CompilationUnit &compUnit,
                       HIRContext &ctx) {
  return HIRBuilder(ctx).build(compUnit);
}
