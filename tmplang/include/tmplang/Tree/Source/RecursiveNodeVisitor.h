#ifndef TMPLANG_TREE_SOURCE_RECURSIVENODEVISITOR_H
#define TMPLANG_TREE_SOURCE_RECURSIVENODEVISITOR_H

#include <llvm/Support/Casting.h>
#include <tmplang/Tree/Source/CompilationUnit.h>

namespace tmplang::source {

// Ugly repetitive control-flow code
#define TRY_TO(Call)                                                           \
  if (!getDerived().Call) {                                                    \
    return false;                                                              \
  }

template <typename Derived> class RecursiveASTVisitor {
public:
  // TODO: Define nodes on .def file to use macro magic
  bool traverseNode(const Node &node) {
    switch (node.getKind()) {
    case Node::Kind::CompilationUnit:
      return getDerived().traverseCompilationUnit(
          *cast<CompilationUnit>(&node));
    case Node::Kind::FuncDecl:
      return getDerived().traverseFuncDecl(*cast<FunctionDecl>(&node));
    case Node::Kind::ParamDecl:
      return getDerived().traverseParamDecl(*cast<ParamDecl>(&node));
    }
    llvm_unreachable("All cases are handled");
  }

  // TODO: Define nodes on .def file to use macro magic
  bool visitNode(const Node &node) {
    switch (node.getKind()) {
    case Node::Kind::CompilationUnit:
      return getDerived().visitCompilationUnit(*cast<CompilationUnit>(&node));
    case Node::Kind::FuncDecl:
      return getDerived().visitFunctionDecl(*cast<FunctionDecl>(&node));
    case Node::Kind::ParamDecl:
      return getDerived().visitParamDecl(*cast<ParamDecl>(&node));
    }
    llvm_unreachable("All cases are handled");
  }

protected:
  //=--------------------------------------------------------------------------=//
  // Begin visit functions
  //=--------------------------------------------------------------------------=//
  // TODO: Define nodes on .def file to use macro magic
  bool visitCompilationUnit(const CompilationUnit &) { return true; }
  bool visitFunctionDecl(const FunctionDecl &) { return true; }
  bool visitParamDecl(const ParamDecl &) { return true; }
  //=--------------------------------------------------------------------------=//
  // End visit functions
  //=--------------------------------------------------------------------------=//

  //=--------------------------------------------------------------------------=//
  // Begin recursive traversal functions
  //=--------------------------------------------------------------------------=//
  bool traverseCompilationUnit(const CompilationUnit &compilationUnit) {
    TRY_TO(visitNode(compilationUnit));
    for (const FunctionDecl &function : compilationUnit.getFunctionDecls()) {
      TRY_TO(traverseNode(function));
    }
    return true;
  }
  bool traverseFuncDecl(const FunctionDecl &funcDecl) {
    TRY_TO(visitNode(funcDecl));
    for (const ParamDecl &param : funcDecl.getParams()) {
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

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_RECURSIVENODEVISITOR_H
