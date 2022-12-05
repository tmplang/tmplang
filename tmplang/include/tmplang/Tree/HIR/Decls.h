#ifndef TMPLANG_TREE_HIR_DECLS_H
#define TMPLANG_TREE_HIR_DECLS_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/Tree/HIR/Decl.h>

namespace tmplang::source {
class Node;
} // namespace tmplang::source

namespace tmplang::hir {

class Type;

class ParamDecl final : public Decl {
public:
  const Type &getType() const { return ParamType; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::ParamDecl;
  }

  explicit ParamDecl(const source::Node &srcNode, StringRef name,
                     const Type &paramType)
      : Decl(Node::Kind::ParamDecl, srcNode, name), ParamType(paramType) {}
  virtual ~ParamDecl() = default;

private:
  const Type &ParamType;
};

class FunctionDecl final : public Decl {
public:
  enum FunctionKind {
    proc = 0, // Pure function, does not modify any state
    fn        // non-pure function
  };

  FunctionKind getFunctionKind() const { return FuncKind; }
  const Type &getReturnType() const { return ReturnType; }
  ArrayRef<ParamDecl> getParams() const { return Params; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::FuncDecl;
  }

  explicit FunctionDecl(const source::Node &srcNode, StringRef name,
                        FunctionKind kind, const Type &returnType,
                        std::vector<ParamDecl> params)
      : Decl(Node::Kind::FuncDecl, srcNode, name), FuncKind(kind),
        ReturnType(returnType), Params(std::move(params)) {}
  virtual ~FunctionDecl() = default;

private:
  FunctionKind FuncKind;
  const Type &ReturnType;
  std::vector<ParamDecl> Params;
};

StringLiteral ToString(FunctionDecl::FunctionKind kind);

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_DECLS_H
