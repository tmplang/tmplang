#ifndef TMPLANG_LOWERING_TRANSFORMS_PASSES_H
#define TMPLANG_LOWERING_TRANSFORMS_PASSES_H

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>
#include <tmplang/Lowering/Dialect/IR/Dialect.h>

#include <memory>

namespace tmplang {

//===----------------------------------------------------------------------===//
// Passes defined in Passes.td
//===----------------------------------------------------------------------===//
#define GEN_PASS_DECL
#include <tmplang/Lowering/Transforms/Passes.h.inc>

std::unique_ptr<mlir::Pass> createRemoveUnreachableOpsAfterReturnPass();

#define GEN_PASS_REGISTRATION
#include <tmplang/Lowering/Transforms/Passes.h.inc>

} // namespace tmplang

#endif // TMPLANG_LOWERING_TRANSFORMS_PASSES_H
