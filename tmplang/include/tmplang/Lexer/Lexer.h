#ifndef TMPLANG_LEXER_LEXER_H
#define TMPLANG_LEXER_LEXER_H

#include <tmplang/Lexer/Token.h>

namespace tmplang {

class Lexer {
public:
  Lexer(llvm::StringRef input);

  const Token &getPrevToken() const;
  const Token &getCurrentToken() const;
  const Token &peakNextToken() const;

  Token next();

  struct LexerState {
    LexerState(llvm::StringRef);

    void advance(unsigned nChars = 1);
    void consumeUntilEOLOrEOF();

    llvm::StringRef CurrentInput;
    SourceLocation CurrentLocation;

    Token PrevToken;
    Token CurrentToken;
    Token NextToken;
  };

private:
  Token nextImpl();

  LexerState State;
  // Detected End of Line
  llvm::StringRef DetectedEOL;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_LEXER_H
