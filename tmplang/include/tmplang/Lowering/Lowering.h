#ifndef TMPLANG_LOWERING_LOWERING_H
#define TMPLANG_LOWERING_LOWERING_H

#include <llvm/IR/Module.h>

namespace tmplang {
class SourceManager;
namespace hir {
class CompilationUnit;
} // namespace hir

enum class MLIRPrintingOpsCfg {
  None = 0,
  Lowering = 1 << 0,
  Optimization = 1 << 1,
  Translation = 1 << 2,
  LLVM = 1 << 3,
  Location = 1 << 4,
  All = Lowering | Optimization | Translation | LLVM | Location,
  LLVM_MARK_AS_BITMASK_ENUM(Location)
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Lower the compilation unit to MLIR
std::unique_ptr<llvm::Module>
Lower(hir::CompilationUnit &compUnit, llvm::LLVMContext &llvmCtx, const SourceManager &sm,
      const MLIRPrintingOpsCfg printingCfg = MLIRPrintingOpsCfg::None);

} // namespace tmplang

#endif // TMPLANG_LOWERING_LOWERING_H
