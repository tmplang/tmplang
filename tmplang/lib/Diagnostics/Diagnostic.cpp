#include <tmplang/Diagnostics/Diagnostic.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Support/SourceManager.h>

#include <cmath>

using namespace tmplang;

llvm::StringLiteral tmplang::ToString(DiagnosticSeverity sev) {
  switch (sev) {
  case DiagnosticSeverity::Warning:
    return "warning";
  case DiagnosticSeverity::Error:
    return "error";
  }
  llvm_unreachable("All cases covered");
}

namespace {

[[maybe_unused]] static constexpr llvm::raw_ostream::Colors RED =
    llvm::raw_ostream::RED;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors WHITE =
    llvm::raw_ostream::WHITE;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors YELLOW =
    llvm::raw_ostream::YELLOW;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors GREEN =
    llvm::raw_ostream::GREEN;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors CYAN =
    llvm::raw_ostream::CYAN;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors BLACK =
    llvm::raw_ostream::BLACK;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors BLUE =
    llvm::raw_ostream::BLUE;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors MAGENTA =
    llvm::raw_ostream::MAGENTA;

} // namespace

DiagnosticSeverity Diagnostic::getSeverity() const {
  return DiagnosticMessages[static_cast<unsigned>(Id)].Sev;
}
llvm::StringRef Diagnostic::getMessage() const {
  return DiagnosticMessages[static_cast<unsigned>(Id)].Msg;
}

void Diagnostic::print(llvm::raw_ostream &out, const SourceManager &sm) const {
  printSeverity(out);
  out << ": ";
  printSummary(out);
  out << "\n";
  printLocation(out, sm);
  out << "\n";
  printContext(out, sm);
  out << "\n";
}

void Diagnostic::printSeverity(llvm::raw_ostream &out) const {
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

void Diagnostic::printSummary(llvm::raw_ostream &out) const {
  out << getMessage();
}

static int GetSpacesNeededForLocation(const DiagnosticSeverity sev) {
  return ToString(sev).size() - llvm::StringRef("at").size();
}

void Diagnostic::printLocation(llvm::raw_ostream &out,
                               const SourceManager &sm) const {
  const int neededSpaces = GetSpacesNeededForLocation(getSeverity());
  assert(neededSpaces >= 0 && "Are we using a severity msg lower than 'at'?");

  out.indent(neededSpaces).changeColor(GREEN, /*Bold=*/true) << "at";

  const LineAndColumn lineAndColumn = sm.getLineAndColumn(LocSpan.Start);
  out.resetColor() << ": " << sm.getFilePath() << ':' << lineAndColumn.Line
                   << ':' << lineAndColumn.Column << '\n';
}

static void
PrintContextLine(llvm::raw_ostream &out, llvm::StringRef lhs,
                 llvm::StringRef rhs = "",
                 llvm::Optional<llvm::raw_ostream::Colors> color = llvm::None) {
  out << ' ' << lhs << ' ';
  out.changeColor(BLUE, /*Bold=*/true) << '|';
  out.resetColor() << ' ';

  if (color) {
    out.changeColor(*color);
  }
  out << rhs;
  out.resetColor() << '\n';
}

namespace {

enum class SubscriptPosition {
  First,
  Middle,
  Last,
};

} // namespace

static llvm::SmallString<80>
GetSubscriptForContext(const unsigned lineWidth, const LineAndColumn start,
                       const LineAndColumn end,
                       const SubscriptPosition subsPos) {
  llvm::SmallString<80> result;

  const char caret = '^';
  const char space = ' ';
  const char tilde = '~';

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

void Diagnostic::printContext(llvm::raw_ostream &out,
                              const SourceManager &sm) const {
  llvm::SmallVector<llvm::StringRef, 2> lines;
  sm.getLines(LocSpan, lines);

  const LineAndColumn lineAndColStart = sm.getLineAndColumn(LocSpan.Start);
  const LineAndColumn lineAndColEnd = sm.getLineAndColumn(LocSpan.End);

  const unsigned biggestNumber = lineAndColStart.Line + lines.size() - 1;
  const std::string spaces(std::log10(biggestNumber) + 1, ' ');

  PrintContextLine(out, spaces);

  for (auto &lineAndIdx : llvm::enumerate(lines)) {
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
