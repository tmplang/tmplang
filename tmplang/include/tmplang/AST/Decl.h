#ifndef TMPLANG_AST_DECL_H
#define TMPLANG_AST_DECL_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/AST/Node.h>

namespace tmplang::hir {

class Decl : public Node {
public:
  llvm::StringRef getName() const { return Name; }

  static bool classof(Node *node) {
    return node->getKind() == Node::Kind::FuncDecl ||
           node->getKind() == Node::Kind::ParamDecl;
  }

  virtual ~Decl() = default;

protected:
  explicit Decl(Node::Kind k, llvm::StringRef name) : Node(k), Name(name) {}

private:
  /// All Decls have a name
  llvm::SmallString<32> Name;
};

} // namespace tmplang::hir

#endif // TMPLANG_AST_DECL_H
