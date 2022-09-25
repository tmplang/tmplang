#ifndef TMPLANG_AST_TYPE_H
#define TMPLANG_AST_TYPE_H

namespace tmplang {

class Type {
public:
  enum Kind {
    K_Builtin,
    K_Path // Path to a type name
  };

  Kind getKind() const { return TKind; }

protected:
  explicit Type(Kind kind) : TKind(kind) {}
  virtual ~Type() = default;

private:
  Kind TKind;
};

} // namespace tmplang

#endif // TMPLANG_AST_TYPE_H
