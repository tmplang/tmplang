#ifndef TMPLANG_LEXER_TOKEN_H
#define TMPLANG_LEXER_TOKEN_H

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/SourceLocation.h>

namespace tmplang {

class SourceManager;

enum TokenKind {
  TK_EOF,
  TK_Unknown,

  TK_IntegerNumber,
  TK_Identifier,

  // Operators
  TK_Eq,

  // Delimiters
  TK_LParentheses,
  TK_RParentheses,
  TK_LKeyBracket,
  TK_RKeyBracket,
  TK_Comma,
  TK_Colon,
  TK_Semicolon,
  TK_RArrow,
  TK_Dot,
  TK_Underscore,

  // Keywords
  TK_Data,
  TK_FnType,
  TK_Match,
  TK_Union,
  TK_ProcType,
  TK_Otherwise,
  TK_Ret,
};
StringLiteral ToString(TokenKind tk);
raw_ostream &operator<<(raw_ostream &out, TokenKind k);

/// Basic unit of processed information from the source code. Each call to
/// the lexer produces one Token.
struct Token {
  Token(TokenKind kind, SourceLocation start, SourceLocation end,
        bool isRecovery = false)
      : Kind(kind), IsErrorRecoveryToken(isRecovery), SrcLocSpan{start, end} {
    assert(!llvm::is_contained({TK_Identifier, TK_IntegerNumber}, kind) &&
           "Invalid constructor for identifiers or numbers");
  }
  Token(StringRef lexeme, TokenKind kind, SourceLocation start,
        SourceLocation end, bool isRecovery = false)
      : Kind(kind), IsErrorRecoveryToken(isRecovery), SrcLocSpan{start, end},
        Lexeme(lexeme) {
    assert(!lexeme.empty());
    assert(llvm::is_contained({TK_Identifier, TK_IntegerNumber}, kind));
  }
  Token() : Kind(TK_Unknown) {}

  bool operator==(const Token &other) const = default;

  void print(raw_ostream &out, const SourceManager &sm) const;
  void dump(const SourceManager &sm) const;

  StringRef getLexeme() const {
    return Kind == TK_Identifier || Kind == TK_IntegerNumber ? Lexeme
                                                             : ToString(Kind);
  }

  // TODO: Extend to use llvm::APInt
  int32_t getNumber() const {
    assert(Kind == TK_IntegerNumber);

    int32_t num = 0;
    for (const char c : getLexeme()) {
      if (c == '_') {
        continue;
      }
      assert(llvm::is_contained("0123456789", c) && "Expected a number");

      num *= 10;
      num += c - '0';
    }
    return num;
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

protected:
  TokenKind getKind() const { return Kind; }

private:
  TokenKind Kind;
  bool IsErrorRecoveryToken = false;
  SourceLocationSpan SrcLocSpan;
  /// Since we keep open the file, storing a reference to the source is valid
  StringRef Lexeme;
};

/// Represents a token with one of the specified kinds
template <TokenKind... Kinds> struct SpecificToken : public Token {
  SpecificToken(Token &&tk) : Token(std::forward<Token>(tk)) {
    assert(llvm::is_contained({Kinds...}, getKind()));
  }
};

} // namespace tmplang

#endif // TMPLANG_LEXER_TOKEN_H
