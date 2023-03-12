#include <mlir/IR/Visitors.h>
#include <mlir/Transforms/Passes.h>
#include <tmplang/Lowering/Dialect/IR/Ops.h>
#include <tmplang/Lowering/Transforms/Passes.h>

namespace tmplang {
#define GEN_PASS_DEF_REMOVEUNREACHABLEOPSAFTERRETURN
#include <tmplang/Lowering/Transforms/Passes.h.inc>
} // namespace tmplang

using namespace mlir;

namespace {

struct RemoveUnreachableOpsAfterReturn
    : public tmplang::impl::RemoveUnreachableOpsAfterReturnBase<
          RemoveUnreachableOpsAfterReturn> {
  RemoveUnreachableOpsAfterReturn() = default;

  void runOnOperation() override {
    SmallVector<tmplang::ReturnOp> rets;
    getOperation()->walk([&](tmplang::ReturnOp retOp) {
      rets.push_back(retOp);
      return mlir::WalkResult::advance();
    });

    while (!rets.empty()) {
      tmplang::ReturnOp ret = rets.pop_back_val();

      auto *nextNode = ret->getNextNode();
      while (nextNode) {
        nextNode->dropAllUses();
        nextNode->erase();
        nextNode = ret->getNextNode();
      }
    }
  }
};

} // namespace

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> tmplang::createRemoveUnreachableOpsAfterReturnPass() {
  return std::make_unique<RemoveUnreachableOpsAfterReturn>();
}
