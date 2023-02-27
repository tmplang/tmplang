#ifndef TMPLANG_TREE_HIR_TYPE_H
#define TMPLANG_TREE_HIR_TYPE_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang::hir {

class Type {
public:
  enum Kind { K_Builtin, K_Tuple, K_Subprogram, K_Data, K_Union };

  Kind getKind() const { return TKind; }

  void print(llvm::raw_ostream &) const;
  void dump() const;

protected:
  explicit Type(Kind kind) : TKind(kind) {}
  virtual ~Type() = default;

private:
  Kind TKind;
};

} // namespace tmplang::hir

#endif // TMPLANG_TREE_HIR_TYPE_H
