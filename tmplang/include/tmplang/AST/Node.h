#ifndef TMPLANG_AST_NODE_H
#define TMPLANG_AST_NODE_H

namespace tmplang::hir {

class Node {
public:
  enum class Kind {
    CompilationUnit = 0,
    FuncDecl,
    ParamDecl,
  };

  Kind getKind() const { return NodeKind; }

protected:
  explicit Node(Kind k) : NodeKind(k) {}
  virtual ~Node() = default;

private:
  Kind NodeKind;
};

} // namespace tmplang::hir

#endif // TMPLANG_AST_NODE_H
