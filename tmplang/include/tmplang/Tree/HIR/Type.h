#ifndef TMPLANG_TREE_HIR_TYPE_H
#define TMPLANG_TREE_HIR_TYPE_H

namespace tmplang::hir {

class Type {
public:
  enum Kind { K_Builtin };

  Kind getKind() const { return TKind; }

protected:
  explicit Type(Kind kind) : TKind(kind) {}
  virtual ~Type() = default;

private:
  Kind TKind;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_TYPE_H
