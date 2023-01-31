#include <tmplang/Lowering/Lowering.h>

#include <llvm/Support/Debug.h>
#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lowering/Conversion/TmplangToArithmetic/TmplangToArithmetic.h>
#include <tmplang/Lowering/Conversion/TmplangToFunc/TmplangToFunc.h>
#include <tmplang/Lowering/Conversion/TmplangToLLVM/TmplangToLLVM.h>
#include <tmplang/Lowering/Dialect/IR/Ops.h>
#include <tmplang/Lowering/Dialect/IR/Types.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/CompilationUnit.h>
#include <tmplang/Tree/HIR/Decls.h>
#include <tmplang/Tree/HIR/Types.h>

#include <memory>

using namespace tmplang;

namespace {

class MLIRBuilder {
public:
  MLIRBuilder(mlir::MLIRContext &ctx, const SourceManager &sm)
      : Ctx(ctx), SM(sm),
        FilePath(mlir::StringAttr::get(&ctx, sm.getFilePath())), B(&Ctx) {}

  mlir::ModuleOp build(const hir::CompilationUnit &compUnit) {
    auto mod = B.create<mlir::ModuleOp>(B.getUnknownLoc(), SM.getFileName());
    for (const std::unique_ptr<hir::Decl> &decl : compUnit.getTopLevelDecls()) {
      if (auto *subprog = dyn_cast<hir::SubprogramDecl>(&*decl)) {
        // SubprogramDecl is the only declaration that can generate code
        B.setInsertionPointToEnd(mod.getBody());
        build(*subprog);
      }
    }
    return mod;
  }

private:
  void build(const hir::SubprogramDecl &subprog) {
    DoesThisFuncReturnInAllPaths = false;

    mlir::FunctionType functionTy = get(subprog.getType());

    auto subprogramOp = B.create<SubprogramOp>(
        getLocation(subprog), subprog.getName(), functionTy,
        mlir::SymbolTable::Visibility::Private);

    // For each param, store for their hir::Symbol, its corresponding
    // mlir::Value
    for (auto hirParamAndMLIRParam : llvm::zip(
             subprog.getParams(), subprogramOp.getBody().getArguments())) {
      auto &[hirParam, mlirParam] = hirParamAndMLIRParam;
      SymbolToValueMap[&hirParam.getSymbol()] = mlirParam;
    }

    B.setInsertionPointToEnd(&subprogramOp.getBody().getBlocks().front());

    for (auto &expr : subprog.getBody()) {
      get(*expr);
    }

    if (!DoesThisFuncReturnInAllPaths) {
      auto *tuple =
          dyn_cast<hir::TupleType>(&subprog.getType().getReturnType());
      if (tuple && tuple->isUnit()) {
        auto unkLoc = mlir::UnknownLoc::get(&Ctx);
        auto tupleOp = B.create<TupleOp>(
            unkLoc, B.getTupleType(mlir::TypeRange()), mlir::ValueRange());
        B.create<ReturnOp>(unkLoc, tupleOp);
      } else {
        // This point means function that is not "void" without return
        llvm_unreachable("This should be error'd out in sema");
      }
    }
  }

  mlir::Value get(const hir::ExprIntegerNumber &expr) {
    mlir::Type ty = get(expr.getType());
    mlir::Attribute attr = B.getIntegerAttr(ty, expr.getNumber());

    return B.create<ConstantOp>(getLocation(expr), ty, attr);
  }

  void build(const hir::ExprRet &exprRet) {
    DoesThisFuncReturnInAllPaths = true;

    mlir::Value val;
    if (auto *expr = exprRet.getReturnedExpr()) {
      val = get(*expr);
    }

    auto retOp = B.create<ReturnOp>(getLocation(exprRet), val);

    // Create a new block so potential instructions that will be generated after
    // the return will be generated on dead blocks, since it will contain no
    // predecessors
    B.createBlock(retOp->getBlock()->getParent());
  }

  mlir::Value get(const hir::ExprTuple &expr) {
    SmallVector<mlir::Value, 4> vals;
    llvm::transform(expr.getVals(), std::back_inserter(vals),
                    [&](const std::unique_ptr<hir::Expr> &hirExpr) {
                      return get(*hirExpr);
                    });

    return B.create<TupleOp>(getLocation(expr), get(expr.getType()), vals);
  }

  mlir::Value get(const hir::ExprVarRef &expr) {
    assert(SymbolToValueMap.count(&expr.getSymbol()) &&
           "Symbol must exists on map");
    return SymbolToValueMap[&expr.getSymbol()];
  }

  mlir::Value get(const hir::ExprAggregateDataAccess &expr) {
    auto baseVal = get(expr.getBase());
    return B.create<AggregateDataAccessOp>(getLocation(expr),
                                           get(expr.getType()), baseVal,
                                           B.getIndexAttr(expr.getIdxAccess()));
  }

  mlir::Value get(const hir::Expr &expr) {
    switch (expr.getKind()) {
    case hir::Node::Kind::ExprIntegerNumber:
      return get(*cast<hir::ExprIntegerNumber>(&expr));
    case hir::Node::Kind::ExprTuple:
      return get(*cast<hir::ExprTuple>(&expr));
    case hir::Node::Kind::ExprVarRef:
      return get(*cast<hir::ExprVarRef>(&expr));
    case hir::Node::Kind::ExprAggregateDataAccess:
      return get(*cast<hir::ExprAggregateDataAccess>(&expr));
    case hir::Node::Kind::ExprRet:
      build(*cast<hir::ExprRet>(&expr));
      // Ret is a terminator expresion
      return {};
    case hir::Node::Kind::CompilationUnit:
    case hir::Node::Kind::SubprogramDecl:
    case hir::Node::Kind::DataFieldDecl:
    case hir::Node::Kind::DataDecl:
    case hir::Node::Kind::ParamDecl:
      llvm_unreachable("None of these are exprs");
    }
  }

  mlir::Type get(const hir::ParamDecl &paramDecl) {
    return get(paramDecl.getType());
  }

  mlir::Type get(const hir::Type &type) {
    switch (type.getKind()) {
    case hir::Type::K_Builtin:
      return get(*cast<hir::BuiltinType>(&type));
    case hir::Type::K_Tuple:
      return get(*cast<hir::TupleType>(&type));
    case hir::Type::K_Subprogram:
      return get(*cast<hir::SubprogramType>(&type));
    case hir::Type::K_Data:
      return get(*cast<hir::DataType>(&type));
    }
    llvm_unreachable("All cases covered");
  }

  mlir::Type get(const hir::BuiltinType &type) {
    switch (type.getBuiltinKind()) {
    case hir::BuiltinType::K_i32:
      return B.getI32Type();
    }
    llvm_unreachable("All cases covered");
  }

  mlir::Type get(const hir::TupleType &type) {
    SmallVector<mlir::Type, 4> types;
    types.reserve(type.getTypes().size());

    transform(type.getTypes(), std::back_inserter(types),
              [&](const hir::Type *type) { return get(*type); });

    return B.getTupleType(types);
  }

  mlir::FunctionType get(const hir::SubprogramType &subprogramTy) {
    std::vector<mlir::Type> inputTypes;
    inputTypes.reserve(subprogramTy.getParamTypes().size());

    transform(subprogramTy.getParamTypes(), std::back_inserter(inputTypes),
              [&](const hir::Type *type) { return get(*type); });

    auto retValTy = get(subprogramTy.getReturnType());
    return B.getFunctionType(inputTypes, retValTy);
  }

  tmplang::DataType get(const hir::DataType &dataTy) {
    SmallVector<mlir::Type, 4> tys;
    transform(dataTy.getFieldsTypes(), std::back_inserter(tys),
              [&](const hir::Type *ty) { return get(*ty); });

    return tmplang::DataType::get(&Ctx, dataTy.getName(), tys);
  }

  mlir::FileLineColLoc getLocation(const hir::Node &node) {
    const LineAndColumn lineAndCol = SM.getLineAndColumn(node.getBeginLoc());
    return mlir::FileLineColLoc::get(FilePath, lineAndCol.Line,
                                     lineAndCol.Column);
  }

private:
  bool DoesThisFuncReturnInAllPaths;
  llvm::DenseMap<const hir::Symbol *, mlir::Value> SymbolToValueMap;

private:
  mlir::MLIRContext &Ctx;
  const SourceManager &SM;
  const mlir::StringAttr FilePath;
  mlir::OpBuilder B;
};

} // namespace

std::unique_ptr<llvm::Module>
tmplang::Lower(hir::CompilationUnit &compUnit, llvm::LLVMContext &llvmCtx,
               const SourceManager &sm, const MLIRPrintingOpsCfg printingCfg) {
  auto ctx = std::make_unique<mlir::MLIRContext>();
  // Load the required dialects (mlir::BuiltinDialect is loaded by default)
  ctx->loadDialect<TmplangDialect>();

  auto mod = MLIRBuilder(*ctx, sm).build(compUnit);

  mlir::OpPrintingFlags flags;
  // So we always get a nicely printed MLIR
  flags.assumeVerified();

  if (static_cast<bool>(printingCfg & MLIRPrintingOpsCfg::Location)) {
    flags.enableDebugInfo();
  }

  // Print the naive module if we got asked for
  if (static_cast<bool>(printingCfg & MLIRPrintingOpsCfg::Lowering)) {
    mod->print(llvm::dbgs(), flags);
  }

  // Load the required dialects for lowering to LLVM
  ctx->loadDialect<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect>();

  // Build pass manager and run pipeline
  mlir::PassManager pm(ctx.get());
  pm.enableVerifier(/*verifyPasses=*/false);

  if (static_cast<bool>(printingCfg & MLIRPrintingOpsCfg::Translation)) {
    // Disable multithreading, otherwise we can't print the IR
    ctx->disableMultithreading();
    // We print the naive module, and we configure to only print after each pass
    mod->print(llvm::dbgs(), flags);
    pm.enableIRPrinting([](mlir::Pass *, mlir::Operation *) { return false; },
                        [](mlir::Pass *, mlir::Operation *) { return true; },
                        false, true, false, llvm::dbgs(), flags);
  }

  // The canonical pass removes unreachable code, and code that is useless to
  // the program execution. Eg:
  //
  //  func foo {
  //    (1, 2); <- the generated code will be removed
  //  }
  pm.addPass(mlir::createCanonicalizerPass());

  // Tmplang specific lowering passes
  pm.addPass(createConvertTmplangToArithmeticPass());
  pm.addPass(createConvertTmplangToFuncPass());
  pm.addPass(createConvertTmplangToLLVMPass());

  // Lower arithmetic pass to LLVM (func is done on TmplangToLLVM)
  pm.addPass(mlir::arith::createConvertArithmeticToLLVMPass());

  if (mlir::failed(pm.run(&*mod))) {
    // FIXME: Use proper diagnostics
    return nullptr;
  }

  // Register LLVM dialect translations
  mlir::registerLLVMDialectTranslation(*ctx);

  // Convert the module to LLVM IR in a new LLVM IR context.
  auto llvmModule = mlir::translateModuleToLLVMIR(mod, llvmCtx);
  if (!llvmModule) {
    // FIXME: Use proper diagnostics
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }

  // Finally, print the LLVM module
  if (static_cast<bool>(printingCfg & MLIRPrintingOpsCfg::LLVM)) {
    llvmModule->dump();
  }

  return llvmModule;
}
