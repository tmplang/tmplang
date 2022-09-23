#include <tmplang/Lexer/Lexer.h>

#include <llvm/ADT/StringSwitch.h>

using namespace tmplang;

namespace {

/// Auxiliary struct to hold the StartLocation and build tokens
struct TokenBuilder {
  explicit TokenBuilder(SourceLocation startLocation)
      : StartLocation(startLocation) {}

  Token buildToken(TokenKind kind, SourceLocation endLocation) {
    if (kind == TokenKind::TK_EOF || kind == TokenKind::TK_Unknown) {
      // If it is a token that adds no characters, do not adjust
      return Token(kind, StartLocation, endLocation);
    }

    // Adjust since tokens starts at 1:1
    return Token(kind, StartLocation,
                 SourceLocation(endLocation.Line, endLocation.Column - 1));
  }

  const SourceLocation StartLocation;
};

} // namespace

Lexer::Lexer(llvm::StringRef input)
    : CurrentInput(input), OriginalInput(input), CurrentLocation(1, 1),
      CurrentToken() {}

static Token NextImpl(llvm::StringRef &currentInput,
                      SourceLocation &currentLocation) {
  TokenBuilder tkBuilder(currentLocation);

  if (currentInput.empty()) {
    return tkBuilder.buildToken(TK_EOF, currentLocation);
  }

  llvm::Optional<TokenKind> simpleTokenMatched;
  switch (currentInput.front()) {
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
    if (currentInput.startswith("->")) {
      simpleTokenMatched = TK_RArrow;
    }
    break;
  case ' ':
  case '\t':
  case '\v':
    currentLocation.Column++;
    currentInput = currentInput.drop_front();
    return NextImpl(currentInput, currentLocation);
  case '\n':
  case '\r':
  case '\f':
    currentLocation.Line++;
    currentLocation.Column = 1;
    currentInput = currentInput.drop_front();
    return NextImpl(currentInput, currentLocation);
  default:
    break;
  }

  if (simpleTokenMatched) {
    const unsigned simpleTokenSize = ToString(*simpleTokenMatched).size();
    currentLocation.Column += simpleTokenSize;
    currentInput = currentInput.drop_front(simpleTokenSize);
    return tkBuilder.buildToken(*simpleTokenMatched, currentLocation);
  }

  // Identifier, ProcType and FnType case. Since all of them are a sequence of
  // alphabetic values we can handle them here together
  llvm::StringRef potentialId =
      currentInput.take_while([](char c) { return std::isalpha(c); });

  if (potentialId.empty()) {
    // Invalid identifier, we don't know what this is
    return tkBuilder.buildToken(TK_Unknown, currentLocation);
  }

  TokenKind tk = llvm::StringSwitch<TokenKind>(potentialId)
                     .Case(ToString(TK_ProcType), TK_ProcType)
                     .Case(ToString(TK_FnType), TK_FnType)
                     .Default(TK_Identifier);

  const unsigned potentialIdSize = potentialId.size();
  currentLocation.Column += potentialIdSize;
  currentInput = currentInput.drop_front(potentialIdSize);
  return tkBuilder.buildToken(tk, currentLocation);
}

Token Lexer::next() {
  return CurrentToken = NextImpl(CurrentInput, CurrentLocation);
}

Token Lexer::prev() const { return CurrentToken; }
