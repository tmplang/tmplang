#ifndef TMPLANG_LOWERING_INITTMPLANG_H
#define TMPLANG_LOWERING_INITTMPLANG_H

#include <mlir/IR/DialectRegistry.h>

// List of Dialects
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <tmplang/Lowering/Dialect/IR/Dialect.h>

// List of Passes
#include <mlir/Transforms/Passes.h>
#include <tmplang/Lowering/Conversion/Passes.h>

namespace tmplang {

/// Register all MLIR passes that tmplang depends on
inline void registerMLIRPassesForTmplang() {
  mlir::registerCanonicalizerPass();
}

/// Register all the dialects used by tmplang
inline void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<tmplang::TmplangDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::LLVM::LLVMDialect>();
}

} // namespace tmplang

#endif // TMPLANG_LOWERING_INITTMPLANG_H
