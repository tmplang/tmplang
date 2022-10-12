#ifndef TMPLANG_LEXER_TOKEN_H
#define TMPLANG_LEXER_TOKEN_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang {

/// Basic struct to hold positions on the source
struct SourceLocation {
  SourceLocation() = default;
  SourceLocation(unsigned line, unsigned column) : Line(line), Column(column) {}

  bool operator==(const SourceLocation &other) const = default;

  unsigned Line = 0;
  unsigned Column = 0;
};

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
  Token(TokenKind kind, SourceLocation start, SourceLocation end)
      : Kind(kind), StartLocation(start), EndLocation(end) {
    assert(kind != TokenKind::TK_Identifier &&
           "Invalid constructor for identifiers");
  }
  Token(llvm::StringRef id, SourceLocation start, SourceLocation end)
      : Kind(TokenKind::TK_Identifier), StartLocation(start), EndLocation(end),
        Lexeme(id) {
    assert(!id.empty());
  }
  Token() : Kind(TK_Unknown) {}

  TokenKind Kind;
  SourceLocation StartLocation;
  SourceLocation EndLocation;

  bool operator==(const Token &other) const = default;

  void print(llvm::raw_ostream &out) const;
  void dump() const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &out, const Token &t);

  llvm::StringRef getLexeme() const {
    assert(Kind == TK_Identifier);
    return Lexeme;
  }

private:
  /// TODO: We could save a lot of bytes here if we keep the files open and
  ///       reused offset-driven locations to retrieve from the source
  llvm::SmallString<32> Lexeme;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_TOKEN_H
