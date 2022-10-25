#ifndef TMPLANG_TREE_HIR_NODE_H
#define TMPLANG_TREE_HIR_NODE_H

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <tmplang/Lexer/SourceLocation.h>
#include <llvm/ADT/BitmaskEnum.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang::source {
class Node;
} // namespace tmplang::source

namespace tmplang::hir {

class Node {
public:
  enum class Kind {
#define HIRNode(KIND) KIND,
#include "../Nodes.def"
#undef HIRNode
  };

  Kind getKind() const { return NodeKind; }

  SourceLocation getBeginLoc() const;
  SourceLocation getEndLoc() const;

  enum PrintConfig {
    None = 0,
    Address = 1 << 0,
    SourceLocation = 1 << 1,
    Color = 1 << 2,
    All = Address | SourceLocation | Color,
    LLVM_MARK_AS_BITMASK_ENUM(Color)
  };

  void print(llvm::raw_ostream &, PrintConfig = PrintConfig::Color) const;
  void dump(PrintConfig = PrintConfig::All) const;

protected:
  explicit Node(Kind k, const source::Node &node)
      : NodeKind(k), SrcNode(node) {}
  virtual ~Node() = default;

private:
  Kind NodeKind;
  // NOTE: This link to the Source Tree means that we have to keep alive the
  // Source Tree as long as the HIR Tree exists. This is cheap in the sense
  // it is just a pointer and only increments each node HIR in 8 bytes.
  // The other possibility is to forward the begin and end SourceLocations to
  // this node. This means in incrementing each node in 2x16 bytes each node.
  const source::Node &SrcNode;
};

inline llvm::StringLiteral ToString(Node::Kind kind) {
  switch (kind) {
#define HIRNode(KIND)                                                          \
  case Node::Kind::KIND:                                                       \
    return #KIND;
#include "../Nodes.def"
#undef HIRNode
  };
  llvm_unreachable("All cases covered");
}

/// Make bitmask operations public
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_NODE_H
