#include <tmplang/Support/SourceManager.h>

#include <llvm/Support/Path.h>
#include <tmplang/Support/FileManager.h>

using namespace tmplang;

SourceManager::SourceManager(const TargetFileEntry &tfe)
    : TargetFile(tfe), MaxSL(tfe.Content->getBufferSize() + /*EOF*/ 1) {
  StringRef content = tfe.Content->getBuffer();

  // Line 1 starts at offset 1, this way we can use 0 to invalid offsets
  OffsetValuesAtStartOfLines.push_back(1);

  for (SourceLocation offset = 1; offset <= content.size(); offset++) {
    if (content[offset - 1] == '\n') {
      OffsetValuesAtStartOfLines.push_back(offset + 1);
    }
  }
}

StringRef SourceManager::getFilePath() const { return TargetFile.RealPathName; }
StringRef SourceManager::getFileName() const {
  return llvm::sys::path::filename(TargetFile.RealPathName);
}

static unsigned GetLineIdx(SourceLocation sl, const SourceLocation maxSL,
                           ArrayRef<SourceLocation> SrcLocValPerLine) {
  assert(sl >= 1 && sl <= maxSL);

  // Return the index of the line that has the value strictly less or equal than
  // the provided. This returns us, in which line the offset lies
  auto *it =
      std::upper_bound(SrcLocValPerLine.begin(), SrcLocValPerLine.end(), sl);
  return std::distance(SrcLocValPerLine.begin(), std::prev(it));
}

StringRef SourceManager::getLine(SourceLocation sl) const {
  SmallVector<StringRef, 1> line;
  getLines({sl, sl}, line);
  assert(line.size() == 1);
  return line.front();
}

void SourceManager::getLines(const SourceLocationSpan span,
                             SmallVectorImpl<StringRef> &lines) const {
  StringRef content = TargetFile.Content->getBuffer();

  const unsigned startIdx =
      GetLineIdx(span.Start, MaxSL, OffsetValuesAtStartOfLines);
  const unsigned endIdx =
      GetLineIdx(span.End, MaxSL, OffsetValuesAtStartOfLines);

  for (unsigned currIdx = startIdx; currIdx <= endIdx; currIdx++) {
    const unsigned startOffset = OffsetValuesAtStartOfLines[currIdx];
    const unsigned endOffset = currIdx + 1 < OffsetValuesAtStartOfLines.size()
                                   ? OffsetValuesAtStartOfLines[currIdx + 1]
                                   : this->MaxSL;
    lines.push_back(
        content.substr(startOffset - /*offset starts at 1*/ 1,
                       (endOffset - startOffset) - /*offset starts at 1*/ 1));
  }
}

LineAndColumn SourceManager::getLineAndColumn(SourceLocation sl) const {
  const unsigned idx = GetLineIdx(sl, MaxSL, OffsetValuesAtStartOfLines);
  return LineAndColumn{.Line = idx + 1,
                       .Column = (sl - OffsetValuesAtStartOfLines[idx]) + 1};
}
