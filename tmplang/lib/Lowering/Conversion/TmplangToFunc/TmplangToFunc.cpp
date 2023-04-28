#include <tmplang/Lowering/Conversion/TmplangToFunc/TmplangToFunc.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <tmplang/Lowering/Dialect/IR/Ops.h>

#include "../PassDetail.h"

using namespace tmplang;

namespace {

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

struct SubprogramOpLowering : public mlir::OpRewritePattern<SubprogramOp> {
  using mlir::OpRewritePattern<SubprogramOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SubprogramOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Create a new non-tmplang function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getName(), op.getFunctionType(), 
        op->getAttrs());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReturnOpLowering : public mlir::OpRewritePattern<ReturnOp> {
  using mlir::OpRewritePattern<ReturnOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReturnOp op, mlir::PatternRewriter &rewriter) const override {
    ReturnOpAdaptor adaptor(op);

    // We lower "tmplang.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
    return mlir::success();
  }
};

} // namespace

////===----------------------------------------------------------------------===//
//// Pass Definition
////===----------------------------------------------------------------------===//

namespace {

struct ConvertTmplangToFuncPass
    : public ConvertTmplangToFuncBase<ConvertTmplangToFuncPass> {
  ConvertTmplangToFuncPass() = default;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    target.addLegalDialect<TmplangDialect, mlir::arith::ArithDialect,
                           mlir::func::FuncDialect>();
    target.addIllegalOp<SubprogramOp, ReturnOp>();

    patterns.add<ReturnOpLowering, SubprogramOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> tmplang::createConvertTmplangToFuncPass() {
  return std::make_unique<ConvertTmplangToFuncPass>();
}
