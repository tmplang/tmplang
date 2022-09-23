#include <tmplang/Lexer/Token.h>

#include <llvm/Support/raw_ostream.h>

using namespace tmplang;

llvm::StringLiteral tmplang::ToString(TokenKind tk) {
  switch (tk) {
  case TK_EOF:
    return "<EOF>";
  case TK_Unknown:
    return "<UNKNOWN>";
  case TK_Identifier:
    return "Ident";
  case TK_LKeyBracket:
    return "{";
  case TK_RKeyBracket:
    return "}";
  case TK_Comma:
    return ",";
  case TK_Semicolon:
    return ";";
  case TK_FnType:
    return "fn";
  case TK_ProcType:
    return "proc";
  case TK_Colon:
    return ":";
  case TK_RArrow:
    return "->";
  }
  llvm_unreachable("Switch covers all cases");
}

llvm::raw_ostream &tmplang::operator<<(llvm::raw_ostream &out, TokenKind k) {
  return out << ToString(k);
}

void Token::print(llvm::raw_ostream &out) const { out << Kind; }
llvm::raw_ostream &tmplang::operator<<(llvm::raw_ostream &out, const Token &t) {
  t.print(out);
  return out;
}
