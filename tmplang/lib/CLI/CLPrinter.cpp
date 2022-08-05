
#include <tmplang/CLI/CLPrinter.h>

#include <llvm/Support/raw_ostream.h>

using namespace tmplang;

llvm::raw_ostream &CLPrinter::outs() { return Outs; }

llvm::raw_ostream &CLPrinter::errs() { return Errs << ExecName << ": error: "; }

llvm::raw_ostream &CLPrinter::warn() {
  return Errs << ExecName << ": warning: ";
}

llvm::StringRef CLPrinter::getExecName() const { return ExecName; }
