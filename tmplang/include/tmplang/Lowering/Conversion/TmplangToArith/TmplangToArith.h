#ifndef TMPLANG_LOWERING_CONVERSION_TMPLANGTOARITH_TMPLANGTOARITH_H
#define TMPLANG_LOWERING_CONVERSION_TMPLANGTOARITH_TMPLANGTOARITH_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tmplang {
std::unique_ptr<mlir::Pass> createConvertTmplangToArithPass();
} // namespace tmplang

#endif // TMPLANG_LOWERING_CONVERSION_TMPLANGTOARITH_TMPLANGTOARITH_H
