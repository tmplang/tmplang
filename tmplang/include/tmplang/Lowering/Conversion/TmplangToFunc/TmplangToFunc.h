#ifndef TMPLANG_LOWERING_CONVERSION_TMPLANGTOFUNC_TMPLANGTOFUNC_H
#define TMPLANG_LOWERING_CONVERSION_TMPLANGTOFUNC_TMPLANGTOFUNC_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tmplang {
std::unique_ptr<mlir::Pass> createConvertTmplangToFuncPass();
} // namespace tmplang

#endif // TMPLANG_LOWERING_CONVERSION_TMPLANGTOFUNC_TMPLANGTOFUNC_H
