#ifndef TMPLANG_AST_NODE_H
#define TMPLANG_AST_NODE_H

namespace tmplang {

class Node {
public:
  enum class Kind {
    FuncDecl = 0,
    ParamDecl,
  };

  Kind getKind() const { return NodeKind; }

protected:
  explicit Node(Kind k) : NodeKind(k) {}
  virtual ~Node() = default;

private:
  Kind NodeKind;
};

} // namespace tmplang

#endif // TMPLANG_AST_NODE_H
