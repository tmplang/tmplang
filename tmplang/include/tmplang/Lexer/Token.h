#ifndef TMPLANG_LEXER_TOKEN_H
#define TMPLANG_LEXER_TOKEN_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/Lexer/SourceLocation.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang {

class SourceManager;

enum TokenKind {
  TK_EOF,
  TK_Unknown,

  TK_Identifier,

  // Delimiters
  TK_LParentheses,
  TK_RParentheses,
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
  Token(TokenKind kind, SourceLocation start, SourceLocation end)
      : Kind(kind), SrcLocSpan{start, end} {
    assert(kind != TokenKind::TK_Identifier &&
           "Invalid constructor for identifiers");
  }
  Token(llvm::StringRef id, SourceLocation start, SourceLocation end)
      : Kind(TokenKind::TK_Identifier), SrcLocSpan{start, end}, Lexeme(id) {
    assert(!id.empty());
  }
  Token() : Kind(TK_Unknown) {}

  bool operator==(const Token &other) const = default;

  void print(llvm::raw_ostream &out, const SourceManager &sm) const;
  void dump(const SourceManager &sm) const;

  llvm::StringRef getLexeme() const {
    assert(Kind == TK_Identifier);
    return Lexeme;
  }

  /// Query functions to know if the Token is or not of some kind/s
  bool is(TokenKind kind) const;
  bool isNot(TokenKind kind) const;
  bool isOneOf(TokenKind f, TokenKind s) const;
  bool isOneOf(TokenKind f, TokenKind s, TokenKind t) const;

  template <typename... TKind_t> bool isOneOf(TKind_t... kinds) const {
    return ((kinds == Kind) || ...);
  }

  SourceLocationSpan getSpan() const { return SrcLocSpan; }

private:
  TokenKind Kind;
  SourceLocationSpan SrcLocSpan;
  /// Since we keep open the file, storing a reference to the source is valid
  llvm::StringRef Lexeme;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_TOKEN_H
