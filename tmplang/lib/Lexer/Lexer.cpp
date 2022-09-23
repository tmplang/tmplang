#include <tmplang/Lexer/Lexer.h>

#include <llvm/ADT/StringSwitch.h>

using namespace tmplang;

Lexer::Lexer(llvm::StringRef input)
    : CurrentInput(input), OriginalInput(input) {}

static Token NextImpl(llvm::StringRef &currentInput) {
  if (currentInput.empty()) {
    return {TK_EOF};
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
  case '\r':
  case '\v':
  case '\f':
    // TODO: increase column counter`
    currentInput = currentInput.drop_front();
    return NextImpl(currentInput);
  case '\n':
    // TODO: increase line counter`
    currentInput = currentInput.drop_front();
    return NextImpl(currentInput);
  default:
    break;
  }

  if (simpleTokenMatched) {
    currentInput =
        currentInput.drop_front(ToString(*simpleTokenMatched).size());
    return {*simpleTokenMatched};
  }

  // Identifier, ProcType and FnType case. Since all of them are a sequence of
  // alphabetic values we can handle them here together
  llvm::StringRef potentialId =
      currentInput.take_while([](char c) { return std::isalpha(c); });

  if (potentialId.empty()) {
    // Invalid identifier, we don't know what this is
    return {TK_Unknown};
  }

  TokenKind tk = llvm::StringSwitch<TokenKind>(potentialId)
                     .Case(ToString(TK_ProcType), TK_ProcType)
                     .Case(ToString(TK_FnType), TK_FnType)
                     .Default(TK_Identifier);

  currentInput = currentInput.drop_front(potentialId.size());
  return {tk};
}

Token Lexer::next() { return CurrentToken = NextImpl(CurrentInput); }

Token Lexer::prev() const { return CurrentToken; }
