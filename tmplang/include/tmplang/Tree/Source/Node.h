#ifndef TMPLANG_TREE_SOURCE_NODE_H
#define TMPLANG_TREE_SOURCE_NODE_H

#include <llvm/ADT/BitmaskEnum.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/ErrorHandling.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/SourceLocation.h>

namespace tmplang {
class SourceManager;
} // namespace tmplang

namespace tmplang::source {

class Node {
public:
  enum class Kind {
#define SourceNode(KIND) KIND,
#include "../Nodes.def"
#undef SourceNode
  };

  virtual tmplang::SourceLocation getBeginLoc() const = 0;
  virtual tmplang::SourceLocation getEndLoc() const = 0;

  enum PrintConfig {
    None = 0,
    Address = 1 << 0,
    SourceLocation = 1 << 1,
    Color = 1 << 2,
    All = Address | SourceLocation | Color,
    LLVM_MARK_AS_BITMASK_ENUM(Color)
  };

  void print(raw_ostream &, const SourceManager &,
             PrintConfig = PrintConfig::Color) const;
  void dump(const SourceManager &, PrintConfig = PrintConfig::All) const;

  Kind getKind() const { return NodeKind; }

protected:
  explicit Node(Kind k) : NodeKind(k) {}
  virtual ~Node() = default;

private:
  Kind NodeKind;
};

inline StringLiteral ToString(Node::Kind kind) {
  switch (kind) {
#define SourceNode(KIND)                                                       \
  case Node::Kind::KIND:                                                       \
    return #KIND;
#include "../Nodes.def"
  };
  llvm_unreachable("All cases covered");
}

/// Make bitmask operations public
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_NODE_H
