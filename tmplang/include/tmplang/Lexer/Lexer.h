#ifndef TMPLANG_LEXER_LEXER_H
#define TMPLANG_LEXER_LEXER_H

#include <tmplang/Lexer/Token.h>

namespace tmplang {

class Lexer {
public:
  Lexer(llvm::StringRef input);

  Token next();
  Token prev() const;

private:
  llvm::StringRef CurrentInput;
  llvm::StringRef OriginalInput;
  Token CurrentToken = {TK_Unknown};
};

} // namespace tmplang

#endif // TMPLANG_LEXER_LEXER_H
