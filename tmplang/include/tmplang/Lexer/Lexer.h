#ifndef TMPLANG_LEXER_LEXER_H
#define TMPLANG_LEXER_LEXER_H

#include <tmplang/Lexer/Token.h>

namespace tmplang {

class Lexer {
public:
  Lexer(llvm::StringRef input);

  Token next();
  Token prev() const;

  struct LexerState {
    LexerState(llvm::StringRef);

    void advance(unsigned nChars = 1);

    llvm::StringRef CurrentInput;
    SourceLocation CurrentLocation;
    Token CurrentToken;
  };

private:
  Token nextImpl();

  LexerState State;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_LEXER_H
