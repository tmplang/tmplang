#ifndef TMPLANG_SUPPORT_SOURCEMANAGER_H
#define TMPLANG_SUPPORT_SOURCEMANAGER_H

#include <llvm/ADT/StringRef.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/SourceLocation.h>

#include <vector>

namespace tmplang {

class TargetFileEntry;

struct LineAndColumn {
  unsigned Line;
  unsigned Column;
  bool operator==(const LineAndColumn &) const = default;
};

/// Contains the target being compiled, and builds a table of offsets based
/// in the newlines of the source code. This table is usefull to reduce the
/// needed quantity of information that must be stored in each node to
/// recover its source location.
/// This class also provides functionality to recover line, columns, filenames,
/// from the encoded source location.
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
