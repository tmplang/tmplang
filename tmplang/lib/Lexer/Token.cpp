#include <tmplang/Lexer/Token.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Support/SourceManager.h>

using namespace tmplang;

StringLiteral tmplang::ToString(TokenKind tk) {
  switch (tk) {
  case TK_EOF:
    return "<EOF>";
  case TK_Unknown:
    return "<UNKNOWN>";
  case TK_Identifier:
    return "Ident";
  case TK_IntegerNumber:
    return "IntegralNumber";
  case TK_Eq:
    return "=";
  case TK_Ret:
    return "ret";
  case TK_LParentheses:
    return "(";
  case TK_RParentheses:
    return ")";
  case TK_LKeyBracket:
    return "{";
  case TK_RKeyBracket:
    return "}";
  case TK_Comma:
    return ",";
  case TK_Semicolon:
    return ";";
  case TK_Dot:
    return ".";
  case TK_Otherwise:
    return "otherwise";
  case TK_Underscore:
    return "_";
  case TK_Match:
    return "match";
  case TK_Union:
    return "union";
  case TK_Data:
    return "data";
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

raw_ostream &tmplang::operator<<(raw_ostream &out, TokenKind k) {
  return out << ToString(k);
}

void Token::print(raw_ostream &out, const SourceManager &sm) const {
  const LineAndColumn start = sm.getLineAndColumn(SrcLocSpan.Start);
  const LineAndColumn end = sm.getLineAndColumn(SrcLocSpan.End);
  out << llvm::formatv("['{0}' {1}:{2}-{3}:{4}]", getLexeme(), start.Line,
                       start.Column, end.Line, end.Column);
}

void Token::dump(const SourceManager &sm) const {
  // FIXME: Add our own debug stream
  print(llvm::dbgs(), sm);
}

bool Token::is(TokenKind kind) const { return Kind == kind; }

bool Token::isNot(TokenKind kind) const { return !is(kind); }

bool Token::isOneOf(TokenKind f, TokenKind s) const { return is(f) || is(s); }

bool Token::isOneOf(TokenKind f, TokenKind s, TokenKind t) const {
  return isOneOf(f, s) || is(t);
}
