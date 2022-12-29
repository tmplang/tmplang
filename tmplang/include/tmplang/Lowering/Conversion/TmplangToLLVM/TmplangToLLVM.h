#ifndef TMPLANG_LOWERING_CONVERSION_TMPLANGTOLLVM_TMPLANGTOLLVM_H
#define TMPLANG_LOWERING_CONVERSION_TMPLANGTOLLVM_TMPLANGTOLLVM_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace tmplang {
std::unique_ptr<mlir::Pass> createConvertTmplangToLLVMPass();
} // namespace tmplang

#endif // TMPLANG_LOWERING_CONVERSION_TMPLANGTOLLVM_TMPLANGTOLLVM_H
