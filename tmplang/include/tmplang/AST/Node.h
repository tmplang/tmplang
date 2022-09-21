#ifndef TMPLANG_AST_NODE_H
#define TMPLANG_AST_NODE_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang {

class Node {
public:
  enum class Kind {
    CompilationUnit = 0,
    FuncDecl,
    ParamDecl,
  };

  Kind getKind() const { return NodeKind; }

  void print(llvm::raw_ostream &) const;
  void dump() const;

protected:
  explicit Node(Kind k) : NodeKind(k) {}
  virtual ~Node() = default;

private:
  Kind NodeKind;
};

} // namespace tmplang

#endif // TMPLANG_AST_NODE_H
