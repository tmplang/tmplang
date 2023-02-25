#ifndef TMPLANG_CONVERSION_PASSES
#define TMPLANG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TmplangToArithmetic
//===----------------------------------------------------------------------===//

def ConvertTmplangToArithmetic : Pass<"convert-tmplang-to-arithmetic"> {
  let summary = "Convert Tmplang dialect to Arithmetic dialect";
  let description = [{
    This pass converts supported Tmplang ops to Arithmetic dialect instructions.
  }];
  let constructor = "tmplang::createConvertTmplangToArithPass()";
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

//===----------------------------------------------------------------------===//
// TmplangToFunc
//===----------------------------------------------------------------------===//

def ConvertTmplangToFunc : Pass<"convert-tmplang-to-func"> {
  let summary = "Convert Tmplang dialect to Func dialect";
  let description = [{
    This pass converts supported Tmplang ops to Func dialect instructions.
  }];
  let constructor = "tmplang::createConvertTmplangToFuncPass()";
  let dependentDialects = ["mlir::func::FuncDialect"];
}

//===----------------------------------------------------------------------===//
// TmplangToLLVM
//===----------------------------------------------------------------------===//

def ConvertTmplangToLLVM : Pass<"convert-tmplang-to-llvm"> {
  let summary = "Convert Tmplang dialect to LLVM dialect";
  let description = [{
    This pass converts supported Tmplang ops to LLVM dialect instructions.
  }];
  let constructor = "tmplang::createConvertTmplangToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // TMPLANG_CONVERSION_PASSES
