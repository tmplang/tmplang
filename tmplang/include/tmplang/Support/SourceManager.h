#ifndef TMPLANG_SUPPORT_SOURCEMANAGER_H
#define TMPLANG_SUPPORT_SOURCEMANAGER_H

#include <llvm/ADT/StringRef.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/SourceLocation.h>

namespace tmplang {

class TargetFileEntry;

struct LineAndColumn {
  unsigned Line;
  unsigned Column;
  bool operator==(const LineAndColumn &) const = default;
};

class SourceManager {
public:
  SourceManager(const TargetFileEntry &tfe);

  StringRef getFilePath() const;
  StringRef getFileName() const;

  StringRef getLine(SourceLocation sl) const;
  void getLines(const SourceLocationSpan span,
                SmallVectorImpl<StringRef> &) const;
  LineAndColumn getLineAndColumn(SourceLocation sl) const;

private:
  const TargetFileEntry &TargetFile;
  const SourceLocation MaxSL;
  std::vector<SourceLocation> OffsetValuesAtStartOfLines;
};

} // namespace tmplang

#endif // TMPLANG_SUPPORT_SOURCEMANAGER_H
