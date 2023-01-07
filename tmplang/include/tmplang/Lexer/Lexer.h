#ifndef TMPLANG_LEXER_LEXER_H
#define TMPLANG_LEXER_LEXER_H

#include <tmplang/Lexer/Token.h>

namespace tmplang {

/// Consumes token by token the source code.
class Lexer {
public:
  Lexer(StringRef input);

  Token next();

  struct LexerState {
    LexerState(StringRef);

    void advance(unsigned nChars = 1);
    void consumeUntilEOLOrEOF();

    StringRef CurrentInput;
    SourceLocation CurrentLocation;
  };

private:
  LexerState State;
  // Detected End of Line
  StringRef DetectedEOL;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_LEXER_H
