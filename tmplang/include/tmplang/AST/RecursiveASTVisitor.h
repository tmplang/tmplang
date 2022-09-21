#ifndef TMPLANG_AST_RECURSIVEASTVISITOR_H
#define TMPLANG_AST_RECURSIVEASTVISITOR_H

#include <llvm/Support/Casting.h>
#include <tmplang/AST/CompilationUnit.h>
#include <tmplang/AST/Decls.h>

namespace tmplang {

// Ugly repetitive control-flow code
#define TRY_TO(Call)                                                           \
  if (!getDerived().Call()) {                                                  \
    return false;                                                              \
  }

template <typename Derived> class RecursiveASTVisitor {
public:
  // TODO: Define nodes on .def file to use macro magic
  bool traverseNode(const Node &node) {
    switch (node.getKind()) {
    case Node::Kind::CompilationUnit:
      return getDerived().traverseCompilationUnit(
          *llvm::cast<CompilationUnit>(&node));
    case Node::Kind::FuncDecl:
      return getDerived().traverseFuncDecl(*llvm::cast<FunctionDecl>(&node));
    case Node::Kind::ParamDecl:
      return getDerived().traverseParamDecl(*llvm::cast<ParamDecl>(&node));
    }
    llvm_unreachable("All cases are handled");
  }

  // TODO: Define nodes on .def file to use macro magic
  bool visitNode(const Node &node) {
    switch (node.getKind()) {
    case Node::Kind::CompilationUnit:
      return getDerived().visitCompilationUnit(
          *llvm::cast<CompilationUnit>(&node));
    case Node::Kind::FuncDecl:
      return getDerived().visitFuncDecl(*llvm::cast<FunctionDecl>(&node));
    case Node::Kind::ParamDecl:
      return getDerived().visitParamDecl(*llvm::cast<ParamDecl>(&node));
    }
    llvm_unreachable("All cases are handled");
  }

protected:
  //=--------------------------------------------------------------------------=//
  // Begin visit functions
  //=--------------------------------------------------------------------------=//
  // TODO: Define nodes on .def file to use macro magic
  bool visitCompilationUnit(const CompilationUnit &) { return true; }
  bool visitFuncDecl(const FunctionDecl &) { return true; }
  bool visitParamDecl(const ParamDecl &) { return true; }
  //=--------------------------------------------------------------------------=//
  // End visit functions
  //=--------------------------------------------------------------------------=//

  //=--------------------------------------------------------------------------=//
  // Begin recursive traversal functions
  //=--------------------------------------------------------------------------=//
  bool traverseCompilationUnit(const CompilationUnit &compilationUnit) {
    TRY_TO(visitNode(compilationUnit));
    for (const auto &decl : compilationUnit.getDecls()) {
      TRY_TO(traverseNode(*decl));
    }
    return true;
  }
  bool traverseFuncDecl(const FunctionDecl &funcDecl) {
    TRY_TO(visitNode(funcDecl));
    for (const auto &param : funcDecl.getParams()) {
      TRY_TO(traverseNode(param));
    }
    return true;
  }
  bool traverseParamDecl(const ParamDecl &paramDecl) {
    TRY_TO(visitNode(paramDecl));
    return true;
  }
  //=--------------------------------------------------------------------------=//
  // End recursive traversal functions
  //=--------------------------------------------------------------------------=//

  Derived &getDerived() { return *static_cast<Derived *>(this); }
};

#undef TRY_TO

} // namespace tmplang

#endif // TMPLANG_AST_RECURSIVEASTVISITOR_H
