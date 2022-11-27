#include <tmplang/Diagnostics/Hint.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/ADT/PrintUtils.h>
#include <tmplang/Support/SourceManager.h>

#include "DiagnosticPrettyPrinter.h"

#include <cmath>
#include <numeric>

using namespace tmplang;

namespace {
struct FormatedHintStrAndSize {
  const unsigned SizeWithNoColors;
  llvm::SmallString<80> FormatedHint;
};
} // namespace

static FormatedHintStrAndSize
GetFormatedHintsAsStr(const bool colors,
                      llvm::ArrayRef<llvm::StringRef> hints) {
  llvm::SmallString<80> smallStr;
  llvm::raw_svector_ostream adaptor(smallStr);

  adaptor.enable_colors(colors);

  if (hints.size() > 1) {
    adaptor.changeColor(YELLOW) << '{';
    adaptor.resetColor();
  }

  printInterleaved(
      hints,
      [&](llvm::StringRef str) {
        adaptor.changeColor(CYAN) << str;
        adaptor.resetColor();
      },
      adaptor);

  if (hints.size() > 1) {
    adaptor.changeColor(YELLOW) << '}';
    adaptor.resetColor();
  }

  // clang-format off
  const unsigned totalSizeWithoutColors =
    (hints.size() > 1 ? 2 : 0) + // left and right key braces
    (hints.size() - 1) * 2 + // commas and spaces
    std::accumulate(hints.begin(), hints.end(), 0,             //
      [](unsigned n, llvm::StringRef rhs){ return n + rhs.size(); }); // content
  // clang-format on

  return {totalSizeWithoutColors, smallStr};
}

void InsertTextAtHint::print(llvm::raw_ostream &out,
                             const SourceManager &sm) const {
  // Line with source code
  const LineAndColumn lineAndColStart = sm.getLineAndColumn(SrcLoc);
  const llvm::StringRef line = sm.getLine(SrcLoc);

  const unsigned adjustedColumn = lineAndColStart.Column - 1;

  llvm::SmallString<80> lineToPrint;
  llvm::raw_svector_ostream adaptor(lineToPrint);

  llvm::StringRef lhsLine = line.substr(0, adjustedColumn);
  adaptor.resetColor() << lhsLine;

  const bool needsLeftSep = !lhsLine.endswith(RequiredLSep);
  if (needsLeftSep) {
    // If it does no end with the required left separator, add it
    adaptor << RequiredLSep;
  }

  const FormatedHintStrAndSize textAndSize =
      GetFormatedHintsAsStr(out.colors_enabled(), Hints);
  adaptor << textAndSize.FormatedHint;

  llvm::StringRef rhsLine = line.substr(adjustedColumn);
  const bool needsRightSep = !rhsLine.startswith(RequiredRSep);
  if (needsRightSep) {
    // If it does no end with the required right separator, add it
    adaptor << RequiredRSep;
  }
  adaptor << rhsLine;

  // Print source code with inserton
  PrintContextLine(out, std::to_string(lineAndColStart.Line), lineToPrint);

  llvm::StringRef subscriptText =
      Hints.size() > 1 ? " try adding one of the following" : "";

  const std::string spaces(std::log10(lineAndColStart.Line) + 1, ' ');
  // Print subscript
  PrintContextLine(
      out, spaces,
      GetSubscriptLine(adjustedColumn +
                           (needsLeftSep ? RequiredLSep.size() : 0),
                       textAndSize.SizeWithNoColors) +
          subscriptText,
      GREEN);
}
