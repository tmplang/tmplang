#include <tmplang/AST/CompilationUnit.h>

#include <llvm/Support/Casting.h>
#include <tmplang/AST/Decls.h>

using namespace tmplang;

const FunctionDecl &CompilationUnit::AddFunctionDecl(FunctionDecl func) {
  OwnedTopLevelDecls.push_back(std::make_unique<FunctionDecl>(std::move(func)));
  return *llvm::cast<FunctionDecl>(&*OwnedTopLevelDecls.back());
}
