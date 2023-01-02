#ifndef TMPLANG_TREE_HIR_MULTIPLENODEVISITOR_H
#define TMPLANG_TREE_HIR_MULTIPLENODEVISITOR_H

#include <llvm/Support/Casting.h>
#include <tmplang/Tree/HIR/CompilationUnit.h>
#include <tmplang/Tree/HIR/Decls.h>
#include <tmplang/Tree/HIR/Exprs.h>

namespace tmplang::hir {

template <typename... Visitors> class MultipleVisitor {
public:
  //=------------------------------------------------------------------------=//
  // Query functions
  //=------------------------------------------------------------------------=//
  const std::tuple<Visitors...> &getVisitors() const { return Vis; }

// Ugly repetitive handling of aborting among all visitors code
#define FORWARD(NodeVisFunc, NodeParam)                                        \
  unsigned idx = 0;                                                            \
  return std::apply(                                                           \
      [&](auto &...vis) {                                                      \
        ((Aborted[idx] ? Aborted[idx++]                                        \
                       : (Aborted[idx++], !vis.NodeVisFunc(NodeParam))),       \
         ...);                                                                 \
        return !llvm::all_of(Aborted, std::identity{});                        \
      },                                                                       \
      Vis);

  bool traverseNode(const Node &node) {
    switch (node.getKind()) {
#define HIRNode(K)                                                             \
  case Node::Kind::K: {                                                        \
    FORWARD(traverse##K, *cast<K>(&node));                                     \
  }
#include "../Nodes.def"
    }
    llvm_unreachable("All cases are handled");
  }

  bool visitNode(const Node &node) {
    switch (node.getKind()) {
#define HIRNode(K)                                                             \
  case Node::Kind::K: {                                                        \
    FORWARD(visit##K, *cast<K>(&node));                                        \
  }
#include "../Nodes.def"
    }
    llvm_unreachable("All cases are handled");
  }

protected:
//=--------------------------------------------------------------------------=//
// Begin visit functions
//=--------------------------------------------------------------------------=//
#define HIRNode(K)                                                             \
  bool visit##K(const K &node) { FORWARD(visit##K, node); }
#include "../Nodes.def"
//=--------------------------------------------------------------------------=//
// End visit functions
//=--------------------------------------------------------------------------=//

//=--------------------------------------------------------------------------=//
// Begin recursive traversal functions
//=--------------------------------------------------------------------------=//
#define HIRNode(K)                                                             \
  bool traverse##K(const K &node) { FORWARD(traverse##K, node); }
#include "../Nodes.def"
  //=--------------------------------------------------------------------------=//
  // End recursive traversal functions
  //=--------------------------------------------------------------------------=//

#undef FORWARD

private:
  static constexpr unsigned NVisitors = sizeof...(Visitors);
  std::array<bool, NVisitors> Aborted;
  std::tuple<Visitors...> Vis;
}; // namespace tmplang::hir

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_MULTIPLENODEVISITOR_H
