#include "DiagnosticPrettyPrinter.h"

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Support/SourceManager.h>

using namespace tmplang;

llvm::SmallString<80> tmplang::GetSubscriptLine(unsigned start, unsigned size) {
  assert(size >= 1);

  llvm::SmallString<80> result;
  llvm::raw_svector_ostream adaptor(result);

  // Fill with spaces until caret
  result.append(start, ' ');

  result += caret;
  result.append(size - 1, tilde);

  return result;
}

void tmplang::PrintContextLine(llvm::raw_ostream &out, llvm::StringRef lhs,
                               const llvm::Twine &rhs,
                               llvm::raw_ostream::Colors rhsColor) {
  out.changeColor(WHITE) << ' ' << lhs << ' ';
  out.changeColor(BLUE, /*Bold=*/true) << '|';
  out.changeColor(rhsColor) << ' ' << rhs << '\n';
  out.resetColor();
}
