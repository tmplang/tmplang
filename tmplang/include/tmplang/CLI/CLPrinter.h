#ifndef TMPLANG_CLI_CLPRINTE_H
#define TMPLANG_CLI_CLPRINTE_H

#include <llvm/ADT/StringRef.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang {

class CLPrinter {
public:
  CLPrinter(llvm::raw_ostream &outs, llvm::raw_ostream &errs,
            llvm::StringRef execName)
      : Outs(outs), Errs(errs), ExecName(execName) {}

  llvm::raw_ostream &outs();
  llvm::raw_ostream &errs();
  llvm::raw_ostream &warn();
  llvm::StringRef getExecName() const;

private:
  llvm::raw_ostream &Outs;
  llvm::raw_ostream &Errs;
  std::string ExecName;
};

} // namespace tmplang

#endif // TMPLANG_CLI_CLPRINTER_H
