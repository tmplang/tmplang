#include <tmplang/AST/CompilationUnit.h>

#include <llvm/Support/Casting.h>
#include <tmplang/AST/Decls.h>

using namespace tmplang::hir;

const FunctionDecl &
CompilationUnit::AddFunctionDecl(llvm::StringRef name,
                                 std::vector<ParamDecl> params,
                                 const Type &returnType) {
  OwnedTopLevelDecls.push_back(
      std::make_unique<FunctionDecl>(name, returnType, std::move(params)));

  return *llvm::cast<FunctionDecl>(&*OwnedTopLevelDecls.back());
}
