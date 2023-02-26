#include <tmplang/Lexer/Lexer.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringSwitch.h>

using namespace tmplang;

namespace {

static constexpr StringLiteral AllNumbers = "0123456789";

/// Auxiliary struct to build tokens while updating the state of the Lexer
struct TokenBuilder {
  explicit TokenBuilder(Lexer::LexerState &state) : State(state) {}

  template <TokenKind kind> Token buildFromLexeme(StringRef lexeme) {
    SourceLocation startLocation = State.CurrentLocation;
    State.advance(lexeme.size());
    return Token(lexeme, kind, startLocation,
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

Lexer::Lexer(StringRef input)
    : State(input), DetectedEOL(State.CurrentInput.detectEOL()) {}

// Matches pattern: [a-zA-Z][a-zA-Z0-9]+;
static StringRef GetIdentifier(StringRef in) {
  bool first = true;
  return in.take_while([&first](char c) {
    if (first) {
      first = false;
      return std::isalpha(c);
    }
    return std::isalnum(c);
  });
}

// Matches pattern: [0-9]([0-9_]*)
// Eg: 1_000_000, 1000, 1_____0
static StringRef GetNumber(StringRef in) {
  return in.take_while(
      [=](char c) { return c == '_' || is_contained(AllNumbers, c); });
}

Token Lexer::next() {
  TokenBuilder tkBuilder(State);

  if (State.CurrentInput.empty()) {
    return tkBuilder.buildToken(TK_EOF);
  }

  // Handle new lines
  if (State.CurrentInput.startswith(DetectedEOL)) {
    State.consumeUntilEOLOrEOF();
    return next();
  }

  std::optional<TokenKind> simpleTokenMatched;
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
  case '=':
    simpleTokenMatched = TK_Eq;
    break;
  case '}':
    simpleTokenMatched = TK_RKeyBracket;
    break;
  case '.':
    simpleTokenMatched = TK_Dot;
    break;
  case '_':
    simpleTokenMatched = TK_Underscore;
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

  // Numbers
  if (is_contained(AllNumbers, State.CurrentInput.front())) {
    const StringRef number = GetNumber(State.CurrentInput);
    return tkBuilder.buildFromLexeme<TK_IntegerNumber>(number);
  }

  // Identifier, ProcType and FnType case. Since all of them are a sequence of
  // alphabetic values we can handle them here together.
  // CAVEAT: This is currently correct because the pattern a identifier matches
  // with, is compatible with "fn", "proc" and "ret":
  StringRef potentialId = GetIdentifier(State.CurrentInput);

  if (potentialId.empty()) {
    // Invalid identifier, we don't know what this is
    return tkBuilder.buildToken(TK_Unknown);
  }

  // FIXME: This switch does not allow using these "identifiers" as valid
  // variable names. Eg: proc ret {} is not valid because ret is not a
  // Identifier token
  TokenKind tk = StringSwitch<TokenKind>(potentialId)
                     .Case(ToString(TK_ProcType), TK_ProcType)
                     .Case(ToString(TK_FnType), TK_FnType)
                     .Case(ToString(TK_Ret), TK_Ret)
                     .Case(ToString(TK_Data), TK_Data)
                     .Case(ToString(TK_Match), TK_Match)
                     .Case(ToString(TK_Union), TK_Union)
                     .Case(ToString(TK_Otherwise), TK_Otherwise)
                     .Default(TK_Identifier);

  if (tk == TK_Identifier) {
    return tkBuilder.buildFromLexeme<TK_Identifier>(potentialId);
  }

  return tkBuilder.buildToken(tk, potentialId.size());
}

Lexer::LexerState::LexerState(StringRef input)
    : CurrentInput(input), CurrentLocation(1) {}

void Lexer::LexerState::advance(unsigned nChars) {
  CurrentInput = CurrentInput.drop_front(nChars);
  CurrentLocation += nChars;
}

void Lexer::LexerState::consumeUntilEOLOrEOF() {
  const size_t it = CurrentInput.find(CurrentInput.detectEOL());
  const size_t toAdvance =
      it == StringRef::npos
          // No more end of lines, so it must be end of file
          ? CurrentInput.size()
          // Just consume all until EOL + EOL size
          : it + CurrentInput.detectEOL().size();
  advance(toAdvance);
}
