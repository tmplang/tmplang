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
    // Adjust since tokens starts at 1:1
    return Token(lexeme, startLocation,
                 SourceLocation(State.CurrentLocation.Line,
                                State.CurrentLocation.Column - 1));
  }

  Token buildToken(TokenKind kind, unsigned nChars = 1) {
    if (kind == TokenKind::TK_EOF || kind == TokenKind::TK_Unknown) {
      // If it is a token that do not add chars, return same location for start
      // and end
      return Token(kind, State.CurrentLocation, State.CurrentLocation);
    }

    SourceLocation startLocation = State.CurrentLocation;
    State.advance(nChars);
    // Adjust since tokens starts at 1:1
    return Token(kind, startLocation,
                 SourceLocation(State.CurrentLocation.Line,
                                State.CurrentLocation.Column - 1));
  }

  Lexer::LexerState &State;
};

} // namespace

Lexer::Lexer(llvm::StringRef input)
    : State(input), DetectedEOL(State.CurrentInput.detectEOL()) {}

Token Lexer::nextImpl() {
  TokenBuilder tkBuilder(State);

  if (State.CurrentInput.empty()) {
    return tkBuilder.buildToken(TK_EOF);
  }

  llvm::Optional<TokenKind> simpleTokenMatched;

  // Handle new lines
  if (State.CurrentInput.startswith(DetectedEOL)) {
    State.consumeUntilEOLOrEOF();
    return nextImpl();
  }

  switch (State.CurrentInput.front()) {
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
    return nextImpl();
  case '/':
    if (!State.CurrentInput.startswith("//")) {
      return tkBuilder.buildToken(TK_Unknown);
    }
    // Simple comment case. Ignore all until EOL or EOF
    State.consumeUntilEOLOrEOF();
    return nextImpl();
  default:
    break;
  }

  if (simpleTokenMatched) {
    return tkBuilder.buildToken(*simpleTokenMatched,
                                ToString(*simpleTokenMatched).size());
  }

  // Identifier, ProcType and FnType case. Since all of them are a sequence of
  // alphabetic values we can handle them here together
  llvm::StringRef potentialId =
      State.CurrentInput.take_while([](char c) { return std::isalpha(c); });

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

Token Lexer::next() { return State.CurrentToken = nextImpl(); }

Token Lexer::prev() const { return State.CurrentToken; }

Lexer::LexerState::LexerState(llvm::StringRef input)
    : CurrentInput(input), CurrentLocation(1, 1), CurrentToken() {}

void Lexer::LexerState::advance(unsigned nChars) {
  static llvm::StringLiteral newLineChars = "\n\r\f";
  bool newLine = false;
  if (nChars == 1) {
    newLine = llvm::is_contained(newLineChars, CurrentInput.front());
  } else {
    assert(!CurrentInput.substr(0, nChars).contains(newLineChars) &&
           "New line characters can not appear on multi char token");
  }

  if (newLine) {
    CurrentLocation.Line++;
    CurrentLocation.Column = 1;
  } else {
    CurrentLocation.Column += nChars;
  }
  CurrentInput = CurrentInput.drop_front(nChars);
}

void Lexer::LexerState::consumeUntilEOLOrEOF() {
  size_t it = CurrentInput.find(CurrentInput.detectEOL());
  if (it == llvm::StringRef::npos) {
    // No more end of lines, so it must be end of file
    CurrentLocation.Column += CurrentInput.size();
    CurrentInput = llvm::StringRef{};
  } else {
    CurrentInput =
        CurrentInput.drop_front(it + CurrentInput.detectEOL().size());
    CurrentLocation.Column = 1;
    CurrentLocation.Line += 1;
  }
}
