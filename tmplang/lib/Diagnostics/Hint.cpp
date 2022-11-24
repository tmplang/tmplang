#include <tmplang/Diagnostics/Hint.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/ADT/PrintUtils.h>
#include <tmplang/Support/SourceManager.h>

#include "DiagnosticPrettyPrinter.h"

#include <cmath>
#include <numeric>

using namespace tmplang;

void PreprendHint::print(llvm::raw_ostream &out,
                         const SourceManager &sm) const {
  llvm::SmallString<80> smallStr;
  llvm::raw_svector_ostream adaptor(smallStr);

  adaptor.enable_colors(true);

  if (Hints.size() > 1) {
    adaptor.changeColor(YELLOW) << '{';
    adaptor.resetColor();
  }

  printInterleaved(
      Hints,
      [&](llvm::StringRef str) {
        adaptor.changeColor(CYAN) << str;
        adaptor.resetColor();
      },
      adaptor);

  if (Hints.size() > 1) {
    adaptor.changeColor(YELLOW) << '}';
    adaptor.resetColor();
  }
  adaptor << ' ';

  // Line with source code
  const LineAndColumn lineAndColStart = sm.getLineAndColumn(SrcLoc);
  PrintContextLine(out, std::to_string(lineAndColStart.Line),
                   smallStr + sm.getLine(SrcLoc));

  // clang-format off
  const unsigned totalSizeWithoutColors = 
    (Hints.size() > 1 ? 2 : 0) + // left and right key braces
    (Hints.size() - 1) * 2 + // commas and spaces
    std::accumulate(Hints.begin(), Hints.end(), 0,             //
      [](unsigned n, llvm::StringRef rhs){ return n + rhs.size(); }); // content
  // clang-format on

  llvm::StringRef extraMsg =
      Hints.size() > 1 ? "try one of the following" : "try this";

  // Line with caret
  const std::string spaces(std::log10(lineAndColStart.Line) + 1, ' ');
  PrintContextLine(out, spaces,
                   GetSubscriptLine(0, totalSizeWithoutColors) + " " +
                       extraMsg, GREEN);
}
