#ifndef TMPLANG_CLI_CLPRINTE_H
#define TMPLANG_CLI_CLPRINTE_H

#include <llvm/ADT/StringRef.h>
#include <tmplang/ADT/LLVM.h>

namespace tmplang {

class CLPrinter {
public:
  CLPrinter(raw_ostream &outs, raw_ostream &errs, StringRef execName)
      : Outs(outs), Errs(errs), ExecName(execName) {}

  raw_ostream &outs();
  raw_ostream &errs();
  raw_ostream &warn();
  StringRef getExecName() const;

private:
  raw_ostream &Outs;
  raw_ostream &Errs;
  std::string ExecName;
};

} // namespace tmplang

#endif // TMPLANG_CLI_CLPRINTER_H
