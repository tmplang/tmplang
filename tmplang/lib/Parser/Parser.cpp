#include <tmplang/Parser/Parser.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

using namespace tmplang;

static void Report(Token got, llvm::ArrayRef<TokenKind> expected,
                   llvm::raw_ostream &outs) {
  llvm::StringRef text = expected.size() > 1
                             ? "any of the following tokens were"
                             : "the following token was";

  auto printCommaSeparated = [](llvm::ArrayRef<TokenKind> tokens,
                                llvm::raw_ostream &outs) {
    for (auto tkAndIdx : llvm::enumerate(tokens)) {
      if (tkAndIdx.index() != 0) {
        outs << ", ";
      }
      outs << "'" << ToString(tkAndIdx.value()) << "'";
    }
  };

  outs << "Unexpected token: '" << ToString(got.Kind) << "' when " << text
       << " expected: ";
  printCommaSeparated(expected, outs);
  outs << "\n";
}

namespace {

class Parser {
public:
  Parser(Lexer &lex) : Lex(lex) {
    // Retrieve first token
    Lex.next();
  }

  bool Start();

private:
  bool FunctionDefinition();
  bool ParamList();
  bool Param();
  bool Block();
  bool FunctionType();
  bool Type();
  bool Identifier();

private:
  bool Match(llvm::ArrayRef<TokenKind> list);
  bool TryMatch(llvm::ArrayRef<TokenKind> list,
                bool consumeTokenIfMatch = false);

  Lexer &Lex;
};

} // namespace

/// Start = Function_Definition*;
///       | EOF;
bool Parser::Start() {
  while (true) {
    if (TryMatch({TokenKind::TK_EOF}, /*consumeTok*/ true)) {
      return true;
    }

    if (!FunctionDefinition()) {
      return false;
    }
  }
}

/// Function_Definition =
///  [1] | Function_Type, Identifier, ":", Param_List, "->", Type, Block
///  [2] | Function_Type, Identifier, ":", Param_List, Block
///  [3] | Function_Type, Identifier, "->", Type, Block
///  [4] | Function_Type, Identifier, Block;
bool Parser::FunctionDefinition() {
  const bool funcAndId = FunctionType() && Identifier();
  if (!funcAndId) {
    return false;
  }

  // [1] && [2]
  if (TryMatch({TK_Colon}, /*consumeTok*/ true)) {
    if (!ParamList()) {
      return false;
    }

    if (TryMatch({TK_RArrow}, /*consumeTok*/ true)) {
      return Type() && Block(); // [1]
    }

    return Block(); // [2]
  }

  if (TryMatch({TK_RArrow}, /*consumeTok*/ true)) {
    return Type() && Block(); // [3]
  }

  return Block(); // [4]
}

/// Param_List = Param (",", Param)*;
bool Parser::ParamList() {
  if (!Param()) {
    return false;
  }

  while (TryMatch({TK_Comma}, /*consumeTok*/ true)) {
    if (!Param()) {
      return false;
    }
  }

  return true;
}

/// Param = Type Identifier;
bool Parser::Param() { return Type() && Identifier(); }

/// Block = "{" "}";
bool Parser::Block() {
  return Match({TK_LKeyBracket}) && Match({TK_RKeyBracket});
}

/// Function_type = "proc" | "fn";
bool Parser::FunctionType() { return Match({TK_ProcType, TK_FnType}); }

/// Type = Identifier;
bool Parser::Type() { return Identifier(); }

/// Identifier = [a-Z]*;
bool Parser::Identifier() { return Match({TK_Identifier}); }

bool Parser::Match(llvm::ArrayRef<TokenKind> list) {
  Token tk = Lex.prev();
  if (!llvm::is_contained(list, tk.Kind)) {
    Report(tk, list, llvm::errs());
    return false;
  }

  Lex.next();
  return true;
}

bool Parser::TryMatch(llvm::ArrayRef<TokenKind> list,
                      bool consumeTokenIfMatch) {
  if (!llvm::is_contained(list, Lex.prev().Kind)) {
    return false;
  }

  if (consumeTokenIfMatch) {
    Lex.next();
  }

  return true;
}

bool tmplang::Parse(tmplang::Lexer &lex) { return Parser(lex).Start(); }
