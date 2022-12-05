#ifndef TMPLANG_LEXER_TOKEN_H
#define TMPLANG_LEXER_TOKEN_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/SourceLocation.h>

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
StringLiteral ToString(TokenKind tk);
raw_ostream &operator<<(raw_ostream &out, TokenKind k);

struct Token {
  Token(TokenKind kind, SourceLocation start, SourceLocation end,
        bool isRecovery = false)
      : Kind(kind), IsErrorRecoveryToken(isRecovery), SrcLocSpan{start, end} {
    assert(kind != TokenKind::TK_Identifier &&
           "Invalid constructor for identifiers");
  }
  Token(StringRef id, SourceLocation start, SourceLocation end,
        bool isRecovery = false)
      : Kind(TokenKind::TK_Identifier),
        IsErrorRecoveryToken(isRecovery), SrcLocSpan{start, end}, Lexeme(id) {
    assert(!id.empty());
  }
  Token() : Kind(TK_Unknown) {}

  bool operator==(const Token &other) const = default;

  void print(raw_ostream &out, const SourceManager &sm) const;
  void dump(const SourceManager &sm) const;

  StringRef getLexeme() const {
    return Kind == TK_Identifier ? Lexeme : ToString(Kind);
  }

  /// Query functions to know if the Token is or not of some kind/s
  bool is(TokenKind kind) const;
  bool isNot(TokenKind kind) const;
  bool isOneOf(TokenKind f, TokenKind s) const;
  bool isOneOf(TokenKind f, TokenKind s, TokenKind t) const;

  template <typename... TKind_t> bool isOneOf(TKind_t... kinds) const {
    return ((kinds == Kind) || ...);
  }

  bool isErrorRecoveryToken() const { return IsErrorRecoveryToken; }
  SourceLocationSpan getSpan() const { return SrcLocSpan; }

private:
  TokenKind Kind;
  bool IsErrorRecoveryToken = false;
  SourceLocationSpan SrcLocSpan;
  /// Since we keep open the file, storing a reference to the source is valid
  StringRef Lexeme;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_TOKEN_H
