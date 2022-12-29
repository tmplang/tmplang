#include <tmplang/Lowering/Conversion/TmplangToArithmetic/TmplangToArithmetic.h>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
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

struct ConstantOpLowering : public mlir::OpRewritePattern<ConstantOp> {
  using mlir::OpRewritePattern<ConstantOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, op.value(),
                                                         op.getType());
    return mlir::success();
  }
};

} // namespace

////===----------------------------------------------------------------------===//
//// Pass Definition
////===----------------------------------------------------------------------===//

namespace {

struct ConvertTmplangToArithmeticPass
    : public ConvertTmplangToArithmeticBase<ConvertTmplangToArithmeticPass> {
  ConvertTmplangToArithmeticPass() = default;

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    target.addLegalDialect<TmplangDialect, mlir::arith::ArithmeticDialect>();
    target.addIllegalOp<ConstantOp>();

    patterns.add<ConstantOpLowering>(&getContext());

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

std::unique_ptr<mlir::Pass> tmplang::createConvertTmplangToArithmeticPass() {
  return std::make_unique<ConvertTmplangToArithmeticPass>();
}
