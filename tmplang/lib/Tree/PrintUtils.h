#ifndef TMPLANG_LIB_TREE_PRINTUTILS_H
#define TMPLANG_LIB_TREE_PRINTUTILS_H

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Lexer/SourceLocation.h>
#include <tmplang/Support/SourceManager.h>

namespace tmplang {

struct TerminalColor {
  raw_ostream::Colors Color;
  bool Bold;
};

class ColorScope {
  raw_ostream &O;
  bool ShowColors;

public:
  ColorScope(raw_ostream &O, bool showColors, TerminalColor color)
      : O(O), ShowColors(showColors) {
    if (ShowColors) {
      O.changeColor(color.Color, color.Bold);
    }
  }
  ~ColorScope() {
    if (ShowColors) {
      O.resetColor();
    }
  }
};

inline void PrintSourceLocation(SourceLocationSpan span, bool showColors,
                                TerminalColor termColor,
                                const SourceManager &sm, raw_ostream &out) {
  ColorScope color(out, showColors, termColor);

  auto getLocStr = [&](SourceLocation loc) -> SmallString<10> {
    switch (loc) {
    case RecoveryLoc:
      return StringRef("[recovery sloc]");
    case InvalidLoc:
      return StringRef("[invalid sloc]");
    default: {
      const LineAndColumn begin = sm.getLineAndColumn(loc);
      return llvm::formatv("{0},{1}", begin.Line, begin.Column);
    }
    }
  };

  out << llvm::formatv(" <{0}-{1}>", getLocStr(span.Start),
                       getLocStr(span.End));
}

inline void PrintPointer(const void *ptr, bool showColors,
                         TerminalColor termColor, raw_ostream &out) {
  ColorScope color(out, showColors, termColor);
  out << ' ' << ptr;
}

} // namespace tmplang

#endif // TMPLANG_LIB_TREE_PRINTUTILS_H
