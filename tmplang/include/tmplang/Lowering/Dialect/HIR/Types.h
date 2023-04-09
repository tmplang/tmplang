#ifndef TMPLANG_LOWERING_DIALECT_IR_TYPES_H
#define TMPLANG_LOWERING_DIALECT_IR_TYPES_H

#include <tmplang/Lowering/Dialect/HIR/Dialect.h>

namespace tmplang {
/// This class represents the base class of all PDL types.
class TmplangType : public mlir::Type {
public:
  using mlir::Type::Type;

  static bool classof(mlir::Type type);
};
} // namespace tmplang

#define GET_TYPEDEF_CLASSES
#include <tmplang/Lowering/Dialect/HIR/TmplangHIROpsTypes.h.inc>

#endif // TMPLANG_LOWERING_DIALECT_IR_TYPES_H
