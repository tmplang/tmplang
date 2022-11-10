#ifndef TMPLANG_TREE_SOURCE_TYPE_H
#define TMPLANG_TREE_SOURCE_TYPE_H

#include <memory>
#include <tmplang/Lexer/SourceLocation.h>
#include <utility>

namespace tmplang::source {

class Type {
public:
  enum Kind {
    NamedType, // eg: i32
    TupleType
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

/// All polymorphic containers should use this type. This is naive approach. We
/// should research whether using an Arena or another approach is ideal.
using RAIIType = std::unique_ptr<Type>;

template <typename T, typename... Args_t>
inline RAIIType make_RAIIType(Args_t &&...args) {
  return std::make_unique<T>(std::forward<Args_t>(args)...);
}

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_TYPE_H
