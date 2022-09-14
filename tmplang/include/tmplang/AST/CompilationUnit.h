#ifndef TMPLANG_PARSER_COMPILATIONUNIT_H
#define TMPLANG_PARSER_COMPILATIONUNIT_H

#include <tmplang/AST/Decls.h>

#include <memory>

namespace tmplang {

class Type;

/// Represents the result of a successfully compiled source file and it is the
/// root node of every AST. This class contains the ownership of every
/// declaration found in the source file.
class CompilationUnit : Node {
public:
  CompilationUnit() : Node(Node::Kind::CompilationUnit) {}

  const FunctionDecl &AddFunctionDecl(llvm::StringRef name,
                                      std::vector<ParamDecl> params,
                                      const Type &returnType);

private:
  /// Owned top level declaration
  std::vector<std::unique_ptr<Decl>> OwnedTopLevelDecls;
};

} // namespace tmplang

#endif // TMPLANG_PARSER_COMPILATIONUNIT_H
