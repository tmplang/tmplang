#include <tmplang/Tree/HIR/Node.h>

#include <tmplang/Tree/Source/Node.h>

using namespace tmplang;
using namespace tmplang::hir;

SourceLocation hir::Node::getBeginLoc() const { return SrcNode.getBeginLoc(); }
SourceLocation hir::Node::getEndLoc() const { return SrcNode.getEndLoc(); }
