
#include <tmplang/CLI/CLPrinter.h>

#include <llvm/Support/raw_ostream.h>

using namespace tmplang;

raw_ostream &CLPrinter::outs() { return Outs; }

raw_ostream &CLPrinter::errs() { return Errs << ExecName << ": error: "; }

raw_ostream &CLPrinter::warn() { return Errs << ExecName << ": warning: "; }

StringRef CLPrinter::getExecName() const { return ExecName; }
