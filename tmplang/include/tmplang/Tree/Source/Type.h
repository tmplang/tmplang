#ifndef TMPLANG_TREE_SOURCE_TYPE_H
#define TMPLANG_TREE_SOURCE_TYPE_H

#include <tmplang/Lexer/SourceLocation.h>

namespace tmplang::source {

class Type {
public:
  enum Kind {
    NamedType // eg: i32, void
  };

  virtual SourceLocation getBeginLoc() const = 0;
  virtual SourceLocation getEndLoc() const = 0;

  Kind getKind() const { return TKind; }

  // Since we are using unique_ptr, the destructor must be public
  virtual ~Type() = default;

protected:
  explicit Type(Kind kind) : TKind(kind) {}

private:
  Kind TKind;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_TYPE_H
