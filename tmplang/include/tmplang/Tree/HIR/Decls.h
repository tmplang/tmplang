#ifndef TMPLANG_TREE_HIR_DECLS_H
#define TMPLANG_TREE_HIR_DECLS_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/Tree/HIR/Decl.h>
#include <tmplang/Tree/HIR/Types.h>

namespace tmplang::source {
class Node;
} // namespace tmplang::source

namespace tmplang::hir {

class Type;
class Expr;

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
                          std::vector<std::unique_ptr<Expr>> exprs);
  virtual ~SubprogramDecl();

private:
  FunctionKind FuncKind;
  std::vector<ParamDecl> Params;
  std::vector<std::unique_ptr<Expr>> Expressions;
};

StringLiteral ToString(SubprogramDecl::FunctionKind kind);

class DataFieldDecl final : public Decl {
public:
  static bool classof(const Node *node) {
    return node->getKind() == Kind::DataFieldDecl;
  }

  explicit DataFieldDecl(const source::Node &srcNode, const Symbol &sym)
      : Decl(Kind::DataFieldDecl, srcNode, sym) {}
  virtual ~DataFieldDecl() = default;
};

class DataDecl final : public Decl {
public:
  static bool classof(const Node *node) {
    return node->getKind() == Kind::DataDecl;
  }

  explicit DataDecl(const source::Node &srcNode, const Symbol &sym,
                    std::vector<DataFieldDecl> fields)
      : Decl(Kind::DataDecl, srcNode, sym), Fields(std::move(fields)) {}
  virtual ~DataDecl() = default;

  ArrayRef<DataFieldDecl> getFields() const { return Fields; }

private:
  std::vector<DataFieldDecl> Fields;
};

class PlaceholderDecl final : public Decl {
public:
  PlaceholderDecl(const source::Node &srcNode, Symbol &sym)
      : Decl(Kind::PlaceholderDecl, srcNode, sym) {}

  static bool classof(const Node *node) {
    return node->getKind() == Kind::PlaceholderDecl;
  }
};

class UnionAlternativeFieldDecl final : public Decl {
public:
  UnionAlternativeFieldDecl(const source::Node &srcNode, const Symbol &sym)
      : Decl(Kind::UnionAlternativeFieldDecl, srcNode, sym) {}

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionAlternativeFieldDecl;
  }
};

class UnionAlternativeDecl final : public Decl {
public:
  UnionAlternativeDecl(const source::Node &srcNode, const Symbol &sym,
                       std::vector<UnionAlternativeFieldDecl> fields)
      : Decl(Kind::UnionAlternativeDecl, srcNode, sym),
        Fields(std::move(fields)) {}

  ArrayRef<UnionAlternativeFieldDecl> getFields() const { return Fields; }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionAlternativeDecl;
  }

private:
  std::vector<UnionAlternativeFieldDecl> Fields;
};

class UnionDecl final : public Decl {
public:
  UnionDecl(const source::Node &srcNode, const Symbol &sym,
            std::vector<UnionAlternativeDecl> alternatives)
      : Decl(Kind::UnionDecl, srcNode, sym),
        Alternatives(std::move(alternatives)) {}

  ArrayRef<UnionAlternativeDecl> getAlternatives() const {
    return Alternatives;
  }

  static bool classof(const Node *node) {
    return node->getKind() == Kind::UnionDecl;
  }

private:
  std::vector<UnionAlternativeDecl> Alternatives;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_DECLS_H
