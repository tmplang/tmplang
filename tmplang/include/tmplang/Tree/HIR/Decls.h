#ifndef TMPLANG_TREE_HIR_DECLS_H
#define TMPLANG_TREE_HIR_DECLS_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/Tree/HIR/Decl.h>

namespace tmplang::hir {

class Type;

class ParamDecl final : public Decl {
public:
  const Type &getType() const { return ParamType; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::ParamDecl;
  }

  explicit ParamDecl(llvm::StringRef name, const Type &paramType)
      : Decl(Node::Kind::ParamDecl, name), ParamType(paramType) {}
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
  llvm::ArrayRef<ParamDecl> getParams() const { return Params; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::FuncDecl;
  }

  explicit FunctionDecl(llvm::StringRef name, FunctionKind kind,
                        const Type &returnType, std::vector<ParamDecl> params)
      : Decl(Node::Kind::FuncDecl, name), FuncKind(kind),
        ReturnType(returnType), Params(std::move(params)) {}
  virtual ~FunctionDecl() = default;

private:
  FunctionKind FuncKind;
  const Type &ReturnType;
  std::vector<ParamDecl> Params;
};

llvm::StringLiteral ToString(FunctionDecl::FunctionKind kind);

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_DECLS_H
