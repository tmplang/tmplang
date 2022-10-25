#ifndef TMPLANG_TREE_HIR_NODE_H
#define TMPLANG_TREE_HIR_NODE_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang::hir {

class Node {
public:
  enum class Kind {
#define HIRNode(KIND) KIND,
#include "../Nodes.def"
#undef HIRNode
  };

  Kind getKind() const { return NodeKind; }

  void print(llvm::raw_ostream &, bool colors = false) const;
  void dump(bool colors = false) const;

protected:
  explicit Node(Kind k) : NodeKind(k) {}
  virtual ~Node() = default;

private:
  Kind NodeKind;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_NODE_H
