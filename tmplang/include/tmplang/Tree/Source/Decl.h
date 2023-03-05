#ifndef TMPLANG_TREE_SOURCE_DECL_H
#define TMPLANG_TREE_SOURCE_DECL_H

#include <llvm/ADT/SmallString.h>
#include <tmplang/Lexer/Token.h>
#include <tmplang/Tree/Source/Node.h>

namespace tmplang::source {

class Decl : public Node {
public:
  virtual ~Decl() = default;

  const auto &getIdentifier() const { return Identifier; }

  /// All declarations have a name
  StringRef getName() const { return Identifier.getLexeme(); }

  static bool classof(const Node *node) {
    return node->getKind() == Node::Kind::SubprogramDecl ||
           node->getKind() == Node::Kind::ParamDecl;
  }

protected:
  explicit Decl(Node::Kind k, SpecificToken<TK_Identifier> id)
      : Node(k), Identifier(std::move(id)) {}

  SpecificToken<TK_Identifier> Identifier;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_DECL_H
