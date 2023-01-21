#ifndef TMPLANG_TREE_HIR_DECLS_H
#define TMPLANG_TREE_HIR_DECLS_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/Tree/HIR/Decl.h>
#include <tmplang/Tree/HIR/Exprs.h>

namespace tmplang::source {
class Node;
} // namespace tmplang::source

namespace tmplang::hir {

class Type;

class ParamDecl final : public Decl {
public:
  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::ParamDecl;
  }

  explicit ParamDecl(const source::Node &srcNode, const Symbol &sym)
      : Decl(Node::Kind::ParamDecl, srcNode, sym) {}
  virtual ~ParamDecl() = default;
};

class SubprogramDecl final : public Decl {
public:
  enum FunctionKind {
    proc = 0, // Pure function, does not modify any state
    fn        // non-pure function
  };

  const SubprogramType &getType() const override;
  FunctionKind getFunctionKind() const { return FuncKind; }
  ArrayRef<ParamDecl> getParams() const { return Params; }
  ArrayRef<std::unique_ptr<Expr>> getBody() const { return Expressions; }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::SubprogramDecl;
  }

  explicit SubprogramDecl(const source::Node &srcNode, const Symbol &sym,
                          FunctionKind kind, std::vector<ParamDecl> params,
                          std::vector<std::unique_ptr<Expr>> exprs)
      : Decl(Node::Kind::SubprogramDecl, srcNode, sym), FuncKind(kind),
        Params(std::move(params)), Expressions(std::move(exprs)) {}
  virtual ~SubprogramDecl() = default;

  SubprogramDecl(SubprogramDecl &&subprogramDecl) = default;

private:
  FunctionKind FuncKind;
  std::vector<ParamDecl> Params;
  std::vector<std::unique_ptr<Expr>> Expressions;
};

StringLiteral ToString(SubprogramDecl::FunctionKind kind);

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_DECLS_H
