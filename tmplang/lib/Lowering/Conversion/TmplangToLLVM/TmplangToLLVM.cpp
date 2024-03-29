#include <tmplang/Lowering/Conversion/TmplangToLLVM/TmplangToLLVM.h>

#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lowering/Dialect/IR/Dialect.h>
#include <tmplang/Lowering/Dialect/IR/Ops.h>
#include <tmplang/Lowering/Dialect/IR/Types.h>

#include "../PassDetail.h"

using namespace tmplang;

namespace {

template <typename TmplangOp>
class TmplangToLLVMConversion : public mlir::ConvertOpToLLVMPattern<TmplangOp> {
public:
  TmplangToLLVMConversion(mlir::LLVMTypeConverter &typeConverter,
                          mlir::PatternBenefit benefit = 1)
      : mlir::ConvertOpToLLVMPattern<TmplangOp>(typeConverter, benefit),
        typeConverter(typeConverter) {}

protected:
  mlir::LLVMTypeConverter &typeConverter;
};

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//
struct TupleOpLowering : public TmplangToLLVMConversion<TupleOp> {
  using TmplangToLLVMConversion<TupleOp>::TmplangToLLVMConversion;

  mlir::LogicalResult
  matchAndRewrite(TupleOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::Location location = op->getLoc();

    mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(
        location, typeConverter.convertType(rewriter.getIndexType()),
        rewriter.getIntegerAttr(
            typeConverter.convertType(rewriter.getIndexType()), 1));

    auto ptrToStructTy =
        typeConverter.convertType(mlir::LLVM::LLVMPointerType::get(
            typeConverter.convertType(op.getType())));

    auto allocaVal =
        rewriter.create<mlir::LLVM::AllocaOp>(location, ptrToStructTy, one);

    storeOperandValuesOnAlloca(op, allocaVal, rewriter);

    auto loadVal = rewriter.create<mlir::LLVM::LoadOp>(location, allocaVal);
    op->replaceAllUsesWith(loadVal);
    rewriter.replaceOp(op, mlir::ValueRange{loadVal});

    return mlir::success();
  }

private:
  void
  storeOperandValuesOnAlloca(TupleOp op, mlir::LLVM::AllocaOp allocaVal,
                             mlir::ConversionPatternRewriter &rewriter) const {
    const mlir::Location location = op->getLoc();

    for (const auto &[idx, operand] : llvm::enumerate(op->getOperands())) {
      auto idxVal = rewriter.create<mlir::LLVM::ConstantOp>(
          location, typeConverter.convertType(rewriter.getIndexType()),
          rewriter.getIntegerAttr(
              typeConverter.convertType(rewriter.getIndexType()), idx));

      auto gepVal = rewriter.create<mlir::LLVM::GEPOp>(
          operand.getLoc(),
          mlir::LLVM::LLVMPointerType::get(
              typeConverter.convertType(operand.getType())),
          allocaVal, mlir::ValueRange{idxVal});

      rewriter.create<mlir::LLVM::StoreOp>(location, operand, gepVal);
    }
  }
};

struct AggregateDataAccessOpLowering
    : public TmplangToLLVMConversion<AggregateDataAccessOp> {
  using TmplangToLLVMConversion<AggregateDataAccessOp>::TmplangToLLVMConversion;

  mlir::LogicalResult
  matchAndRewrite(AggregateDataAccessOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(
        op, adaptor.getOperands()[0], op.getIdx().getSExtValue());
    return mlir::success();
  }
};

struct MatchOpLowering : public TmplangToLLVMConversion<MatchOp> {
  using TmplangToLLVMConversion<MatchOp>::TmplangToLLVMConversion;

  mlir::LogicalResult
  matchAndRewrite(MatchOp matchOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Get the next position of MatchOp so we can split the block at that
    // position. This split allow us to add a block argument corresponding
    // to (all) yield values from all MatchOp branches.
    mlir::Block::iterator nextOp = std::next(rewriter.getInsertionPoint());
    auto *newBlock = rewriter.splitBlock(rewriter.getInsertionBlock(), nextOp);
    newBlock->addArgument(matchOp.getResult().getType(), matchOp->getLoc());

    // Rewrite all yield's for a inconditional branch to the new splited block
    matchOp.getBody().walk([&](tmplang::MatchYieldOp yield) {
      rewriter.setInsertionPoint(yield);
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield, newBlock,
                                                      yield.getResults());
      return mlir::WalkResult::advance();
    });

    // Replace all uses of the match op with the new argument of the block
    matchOp->replaceAllUsesWith(newBlock->getArguments());

    // Insert an inconditional branch to the starting block of the MatchOp,
    // which is the entry point of the MatchOp
    rewriter.setInsertionPointAfter(matchOp);
    rewriter.create<mlir::cf::BranchOp>(matchOp->getLoc(),
                                        &*matchOp.getBody().begin());

    // Inline the whole body region of the MatchOp before the newblock
    rewriter.inlineRegionBefore(matchOp.getBody(), newBlock);

    // Finally, remove the MatchOp
    rewriter.eraseOp(matchOp);

    return mlir::success();
  }
};

} // namespace

////===----------------------------------------------------------------------===//
//// Pass Definition
////===----------------------------------------------------------------------===//

namespace {

struct ConvertTmplangToLLVMPass
    : public ConvertTmplangToLLVMBase<ConvertTmplangToLLVMPass> {
  ConvertTmplangToLLVMPass() = default;

  void runOnOperation() override {
    mlir::LLVMTypeConverter typeConverter(&getContext());
    populateTmplangToLLVMConversionPatterns(getContext(), typeConverter);

    mlir::LLVMConversionTarget target(getContext());
    target
        .addLegalDialect<mlir::LLVM::LLVMDialect, mlir::cf::ControlFlowDialect,
                         mlir::arith::ArithDialect>();
    target
        .addIllegalDialect<mlir::func::FuncDialect, tmplang::TmplangDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    patterns
        .add<AggregateDataAccessOpLowering, TupleOpLowering, MatchOpLowering>(
            typeConverter);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      mlir::Pass::signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Patterns Population
//===----------------------------------------------------------------------===//

void tmplang::populateTmplangToLLVMConversionPatterns(
    mlir::MLIRContext &context, mlir::LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](mlir::TupleType tupleTy) {
    SmallVector<mlir::Type, 4> tys;
    for (auto ty : tupleTy) {
      tys.push_back(typeConverter.convertType(ty));
    }
    return mlir::LLVM::LLVMStructType::getLiteral(&context, tys);
  });
  typeConverter.addConversion([&](tmplang::DataType dataType) {
    SmallVector<mlir::Type, 4> tys;
    for (auto ty : dataType.getTys()) {
      tys.push_back(typeConverter.convertType(ty));
    }
    return mlir::LLVM::LLVMStructType::getNewIdentified(
        &context, dataType.getName(), tys);
  });
}

std::unique_ptr<mlir::Pass> tmplang::createConvertTmplangToLLVMPass() {
  return std::make_unique<ConvertTmplangToLLVMPass>();
}
