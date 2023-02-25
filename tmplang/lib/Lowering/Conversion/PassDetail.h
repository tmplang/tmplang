#ifndef LIB_LOWERING_CONVESION_PASSDETAIL_H
#define LIB_LOWERING_CONVESION_PASSDETAIL_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithDialect;
} // namespace arith

namespace func {
class FuncDialect;
} // namespace func

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

} // namespace mlir

namespace tmplang {
class TmplangDialect;

#define GEN_PASS_CLASSES
#include <tmplang/Lowering/Conversion/Passes.h.inc>

} // namespace tmplang

#endif // LIB_LOWERING_CONVESION_PASSDETAIL_H
