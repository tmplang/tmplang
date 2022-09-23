#ifndef TMPLANG_LEXER_TOKEN_H
#define TMPLANG_LEXER_TOKEN_H

#include <llvm/ADT/StringRef.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang {

enum TokenKind {
  TK_EOF,
  TK_Unknown,

  TK_Identifier,

  // Delimiters
  TK_LKeyBracket,
  TK_RKeyBracket,
  TK_Comma,
  TK_Colon,
  TK_Semicolon,
  TK_RArrow,

  // Keywords
  TK_FnType,
  TK_ProcType,
};
llvm::StringLiteral ToString(TokenKind tk);
llvm::raw_ostream &operator<<(llvm::raw_ostream &out, TokenKind k);

struct Token {
  TokenKind Kind;

  bool operator==(const Token &other) const = default;
  void print(llvm::raw_ostream &out) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &out, const Token &t);
};

} // namespace tmplang

#endif // TMPLANG_LEXER_TOKEN_H
