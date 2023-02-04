#include <tmplang/Lowering/Conversion/TmplangToLLVM/TmplangToLLVM.h>

#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Support/LogicalResult.h>
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

    for (auto &[idx, operand] : llvm::enumerate(op->getOperands())) {
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

} // namespace

////===----------------------------------------------------------------------===//
//// Pass Definition
////===----------------------------------------------------------------------===//

namespace {

struct ConvertTmplangToLLVMPass
    : public ConvertTmplangToLLVMBase<ConvertTmplangToLLVMPass> {
  ConvertTmplangToLLVMPass() = default;

  void runOnOperation() override {
    mlir::LLVMConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    mlir::LLVMTypeConverter converter(&getContext());
    populateTmplangToLLVMConversionPatterns(getContext(), typeConverter);
    // Since we want to lower builtin tuple types, we need to lower func dialect
    // to LLVM along this pass
    mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
    patterns.add<TupleOpLowering>(converter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
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
