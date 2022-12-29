#ifndef TMPLANG_LOWERING_CONVERSION_TMPLANGTOARITHMETIC_TMPLANGTOARITHMETIC_H
#define TMPLANG_LOWERING_CONVERSION_TMPLANGTOARITHMETIC_TMPLANGTOARITHMETIC_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tmplang {
std::unique_ptr<mlir::Pass> createConvertTmplangToArithmeticPass();
} // namespace tmplang

#endif // TMPLANG_LOWERING_CONVERSION_TMPLANGTOARITHMETIC_TMPLANGTOARITHMETIC_H
