#ifndef TMPLANG_AST_DECLS_H
#define TMPLANG_AST_DECLS_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/AST/Decl.h>

namespace tmplang {

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
  const Type &getReturnType() const { return ReturnType; }
  llvm::ArrayRef<ParamDecl> getParams() const { return Params; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::FuncDecl;
  }

  explicit FunctionDecl(llvm::StringRef name, const Type &returnType,
                        std::vector<ParamDecl> params)
      : Decl(Node::Kind::FuncDecl, name), ReturnType(returnType),
        Params(std::move(params)) {}
  virtual ~FunctionDecl() = default;

private:
  const Type &ReturnType;
  std::vector<ParamDecl> Params;
};

} // namespace tmplang

#endif // TMPLANG_AST_DECLS_H
