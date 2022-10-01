#ifndef TMPLANG_TREE_SOURCE_NODE_H
#define TMPLANG_TREE_SOURCE_NODE_H

#include <tmplang/Lexer/SourceLocation.h>

namespace tmplang::source {

class Node {
public:
  enum class Kind {
#define SourceNode(KIND) KIND,
#include "../Nodes.def"
#undef SourceNode
  };

  virtual SourceLocation getBeginLoc() const = 0;
  virtual SourceLocation getEndLoc() const = 0;

  Kind getKind() const { return NodeKind; }

protected:
  explicit Node(Kind k) : NodeKind(k) {}
  virtual ~Node() = default;

private:
  Kind NodeKind;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_NODE_H
