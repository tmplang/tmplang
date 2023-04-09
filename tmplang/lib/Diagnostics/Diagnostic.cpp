#include <tmplang/Diagnostics/Diagnostic.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <tmplang/Diagnostics/Hint.h>
#include <tmplang/Support/SourceManager.h>

#include "DiagnosticPrettyPrinter.h"

#include <cmath>

using namespace tmplang;

StringLiteral tmplang::ToString(DiagnosticSeverity sev) {
  switch (sev) {
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  }
  llvm_unreachable("All cases covered");
}

DiagnosticSeverity Diagnostic::getSeverity() const {
  return DiagnosticMessages[static_cast<unsigned>(Id)].Sev;
}
StringRef Diagnostic::getMessage() const {
  return DiagnosticMessages[static_cast<unsigned>(Id)].Msg;
}

void Diagnostic::print(raw_ostream &out, const SourceManager &sm) const {
  printSeverity(out);
  out << ": ";
  printSummary(out);
  out << "\n";
  printLocation(out, sm);
  printContext(out, sm);
  out << "\n";
  printHint(out, sm);
  out << "\n";
}

void Diagnostic::printSeverity(raw_ostream &out) const {
  const DiagnosticSeverity sev = getSeverity();
  switch (sev) {
  case DiagnosticSeverity::Warning:
    out.changeColor(MAGENTA, /*Bold=*/true);
    break;
  case DiagnosticSeverity::Error:
    out.changeColor(RED, /*Bold=*/true);
    break;
  }

  out << ToString(sev);

  out.resetColor();
}

void Diagnostic::printSummary(raw_ostream &out) const { out << getMessage(); }

static int GetSpacesNeededForLocation(const DiagnosticSeverity sev) {
  return ToString(sev).size() - StringRef("at").size();
}

void Diagnostic::printLocation(raw_ostream &out,
                               const SourceManager &sm) const {
  const int neededSpaces = GetSpacesNeededForLocation(getSeverity());
  assert(neededSpaces >= 0 && "Are we using a severity msg lower than 'at'?");

  out.indent(neededSpaces).changeColor(GREEN, /*Bold=*/true) << "at";

  const LineAndColumn lineAndColumn = sm.getLineAndColumn(LocSpan.Start);
  out.resetColor() << ": " << sm.getFilePath() << ':' << lineAndColumn.Line
                   << ':' << lineAndColumn.Column << '\n';
}

void Diagnostic::printHint(raw_ostream &out, const SourceManager &sm) const {
  if (isa<NoHint>(&H)) {
    return;
  }

  out.changeColor(YELLOW) << "Hint";
  out.resetColor() << ": \n";
  H.print(out, sm);
}

namespace {

enum class SubscriptPosition {
  First,
  Middle,
  Last,
};

} // namespace

static SmallString<80> GetSubscriptForContext(const unsigned lineWidth,
                                              const LineAndColumn start,
                                              const LineAndColumn end,
                                              const SubscriptPosition subsPos) {
  SmallString<80> result;

  switch (subsPos) {
  case SubscriptPosition::First:
    result.append(start.Column - /*offset start at 1*/ 1, space);
    result += caret;
    result.append((start.Line != end.Line ? lineWidth : end.Column) -
                      start.Column,
                  tilde);
    break;
  case SubscriptPosition::Middle:
    result.append(lineWidth, tilde);
    break;
  case SubscriptPosition::Last:
    result.append(end.Column, tilde);
    break;
  }

  return result;
}

static SubscriptPosition GetSubscriptPosition(const unsigned lineIdx,
                                              const unsigned numOfLines) {
  if (lineIdx == 0) {
    return SubscriptPosition::First;
  }

  return lineIdx == numOfLines - 1 ? SubscriptPosition::Last
                                   : SubscriptPosition::Middle;
}

void Diagnostic::printContext(raw_ostream &out, const SourceManager &sm) const {
  SmallVector<StringRef, 2> lines;
  sm.getLines(LocSpan, lines);

  const LineAndColumn lineAndColStart = sm.getLineAndColumn(LocSpan.Start);
  const LineAndColumn lineAndColEnd = sm.getLineAndColumn(LocSpan.End);

  const unsigned biggestNumber = lineAndColStart.Line + lines.size() - 1;
  const std::string spaces(std::log10(biggestNumber) + 1, ' ');

  PrintContextLine(out, spaces);

  for (const auto &lineAndIdx : llvm::enumerate(lines)) {
    auto [idx, line] = lineAndIdx;

    // Line with source code
    PrintContextLine(out, std::to_string(idx + lineAndColStart.Line), line);

    if (!line.empty()) {
      // Line with caret
      PrintContextLine(
          out, spaces,
          GetSubscriptForContext(line.size(), lineAndColStart, lineAndColEnd,
                                 GetSubscriptPosition(idx, lines.size())),
          GREEN);
    }
  }

  PrintContextLine(out, spaces);
}
