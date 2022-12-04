#include <tmplang/Lexer/Lexer.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>

using namespace tmplang;

namespace {

/// Auxiliary struct to build tokens while updating the state of the Lexer
struct TokenBuilder {
  explicit TokenBuilder(Lexer::LexerState &state) : State(state) {}

  Token buildIdentifier(llvm::StringRef lexeme) {
    SourceLocation startLocation = State.CurrentLocation;
    State.advance(lexeme.size());
    return Token(lexeme, startLocation,
                 State.CurrentLocation - /*offset starts at 1*/ 1);
  }

  Token buildToken(TokenKind kind, unsigned nChars = 1) {
    SourceLocation startLocation = State.CurrentLocation;

    if (kind == TokenKind::TK_EOF || kind == TokenKind::TK_Unknown) {
      // If it is a token that do not add chars, return same location for start
      // and end
      State.advance(kind == TK_Unknown ? 1 : 0);
      return Token(kind, startLocation, startLocation);
    }

    State.advance(nChars);
    return Token(kind, startLocation,
                 State.CurrentLocation - /*offset starts at 1*/ 1);
  }

  Lexer::LexerState &State;
};

} // namespace

Lexer::Lexer(llvm::StringRef input)
    : State(input), DetectedEOL(State.CurrentInput.detectEOL()) {}

// Matches pattern: [a-zA-Z][a-zA-Z0-9]+;
static llvm::StringRef GetIdentifier(llvm::StringRef in) {
  bool first = true;
  return in.take_while([&first](char c) {
    if (first) {
      first = false;
      return std::isalpha(c);
    }
    return std::isalnum(c);
  });
}

Token Lexer::next() {
  TokenBuilder tkBuilder(State);

  if (State.CurrentInput.empty()) {
    return tkBuilder.buildToken(TK_EOF);
  }

  llvm::Optional<TokenKind> simpleTokenMatched;

  // Handle new lines
  if (State.CurrentInput.startswith(DetectedEOL)) {
    State.consumeUntilEOLOrEOF();
    return next();
  }

  switch (State.CurrentInput.front()) {
  case '(':
    simpleTokenMatched = TK_LParentheses;
    break;
  case ')':
    simpleTokenMatched = TK_RParentheses;
    break;
  case ';':
    simpleTokenMatched = TK_Semicolon;
    break;
  case ':':
    simpleTokenMatched = TK_Colon;
    break;
  case '{':
    simpleTokenMatched = TK_LKeyBracket;
    break;
  case ',':
    simpleTokenMatched = TK_Comma;
    break;
  case '}':
    simpleTokenMatched = TK_RKeyBracket;
    break;
  case '-':
    if (State.CurrentInput.startswith("->")) {
      simpleTokenMatched = TK_RArrow;
    }
    break;
  case ' ':
  case '\t':
  case '\v':
    State.advance();
    return next();
  case '/':
    if (!State.CurrentInput.startswith("//")) {
      return tkBuilder.buildToken(TK_Unknown);
    }
    // Simple comment case. Ignore all until EOL or EOF
    State.consumeUntilEOLOrEOF();
    return next();
  default:
    break;
  }

  if (simpleTokenMatched) {
    return tkBuilder.buildToken(*simpleTokenMatched,
                                ToString(*simpleTokenMatched).size());
  }

  // Identifier, ProcType and FnType case. Since all of them are a sequence of
  // alphabetic values we can handle them here together.
  // CAVEAT: This is currently correct because the pattern a identifier matches
  // with, is compatible with "fn" and "proc:"
  llvm::StringRef potentialId = GetIdentifier(State.CurrentInput);

  if (potentialId.empty()) {
    // Invalid identifier, we don't know what this is
    return tkBuilder.buildToken(TK_Unknown);
  }

  TokenKind tk = llvm::StringSwitch<TokenKind>(potentialId)
                     .Case(ToString(TK_ProcType), TK_ProcType)
                     .Case(ToString(TK_FnType), TK_FnType)
                     .Default(TK_Identifier);

  if (tk == TK_Identifier) {
    return tkBuilder.buildIdentifier(potentialId);
  }

  return tkBuilder.buildToken(tk, potentialId.size());
}

Lexer::LexerState::LexerState(llvm::StringRef input)
    : CurrentInput(input), CurrentLocation(1) {}

void Lexer::LexerState::advance(unsigned nChars) {
  CurrentInput = CurrentInput.drop_front(nChars);
  CurrentLocation += nChars;
}

void Lexer::LexerState::consumeUntilEOLOrEOF() {
  const size_t it = CurrentInput.find(CurrentInput.detectEOL());
  const size_t toAdvance =
      it == llvm::StringRef::npos
          // No more end of lines, so it must be end of file
          ? CurrentInput.size()
          // Just consume all until EOL + EOL size
          : it + CurrentInput.detectEOL().size();
  advance(toAdvance);
}
