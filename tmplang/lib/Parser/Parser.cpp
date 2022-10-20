#include <tmplang/Parser/Parser.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Tree/Source/Decls.h>
#include <tmplang/Tree/Source/Types.h>

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

struct LexicalScope {
  Token LKeyBracket;
  Token RKeyBracket;
};

class Parser {
public:
  Parser(Lexer &lex) : Lex(lex) {
    // Retrieve first token
    Lex.next();
  }

  llvm::Optional<source::CompilationUnit> Start();

private:
  llvm::Optional<source::FunctionDecl> FunctionDefinition();
  llvm::Optional<source::FunctionDecl::ParamList> ParamList();
  llvm::Optional<source::ParamDecl> Param();
  llvm::Optional<LexicalScope> Block();
  llvm::Optional<Token> FunctionType();
  llvm::Optional<source::NamedType> Type();
  llvm::Optional<Token> Identifier();

private:
  llvm::Optional<Token> Match(llvm::ArrayRef<TokenKind> list);
  llvm::Optional<Token> TryMatch(llvm::ArrayRef<TokenKind> list,
                                 bool consumeTokenIfMatch = false);

  Lexer &Lex;
};

} // namespace

/// Start = Function_Definition*;
///       | EOF;
llvm::Optional<source::CompilationUnit> Parser::Start() {
  source::CompilationUnit compilationUnit;

  while (true) {
    if (TryMatch({TokenKind::TK_EOF}, /*consumeTok*/ true)) {
      return compilationUnit;
    }

    auto func = FunctionDefinition();
    if (!func) {
      return llvm::None;
    }

    compilationUnit.addFunctionDecl(std::move(*func));
  }

  return compilationUnit;
}

/// Function_Definition =
///  [1] | Function_Type, Identifier, ":", Param_List, "->", Type, Block
///  [2] | Function_Type, Identifier, ":", Param_List, Block
///  [3] | Function_Type, Identifier, "->", Type, Block
///  [4] | Function_Type, Identifier, Block;
llvm::Optional<source::FunctionDecl> Parser::FunctionDefinition() {
  auto funcType = FunctionType();
  auto id = Identifier();
  if (!funcType || !id) {
    return llvm::None;
  }

  // [1] && [2]
  if (auto colon = TryMatch({TK_Colon}, /*consumeTok*/ true)) {
    auto paramList = ParamList();
    if (!paramList) {
      return llvm::None;
    }

    if (auto arrow = TryMatch({TK_RArrow}, /*consumeTok*/ true)) {
      // [1]
      auto returnType = Type();
      auto block = Block();
      if (!returnType || !block) {
        return llvm::None;
      }

      return source::FunctionDecl::Create(
          *funcType, *id, *colon, std::move(*paramList),
          source::FunctionDecl::ArrowAndType{*arrow,
                                             source::NamedType(*returnType)},
          block->LKeyBracket, block->RKeyBracket);
    }

    // [2]
    auto block = Block();
    if (!block) {
      return llvm::None;
    }

    return source::FunctionDecl::Create(*funcType, *id, *colon,
                                        std::move(*paramList),
                                        block->LKeyBracket, block->RKeyBracket);
  }

  if (auto arrow = TryMatch({TK_RArrow}, /*consumeTok*/ true)) {
    // [3]
    auto returnType = Type();
    auto block = Block();
    if (!returnType || !block) {
      return llvm::None;
    }

    return source::FunctionDecl::Create(
        *funcType, *id,
        source::FunctionDecl::ArrowAndType{*arrow,
                                           source::NamedType(*returnType)},
        block->LKeyBracket, block->RKeyBracket);
  }

  // [4]
  auto block = Block();
  if (!block) {
    return llvm::None;
  }
  return source::FunctionDecl::Create(*funcType, *id, block->LKeyBracket,
                                      block->RKeyBracket);
}

/// Param_List = Param (",", Param)*;
llvm::Optional<source::FunctionDecl::ParamList> Parser::ParamList() {
  source::FunctionDecl::ParamList paramList;

  auto firstParam = Param();
  if (!firstParam) {
    return llvm::None;
  }

  paramList.ParamList.push_back(std::move(*firstParam));

  while (auto comma = TryMatch({TK_Comma}, /*consumeTok*/ true)) {
    auto param = Param();
    if (!param) {
      return llvm::None;
    }

    paramList.ParamList.push_back(std::move(*param));
    paramList.CommaList.push_back(*comma);
  }

  return paramList;
}

/// Param = Type Identifier;
llvm::Optional<source::ParamDecl> Parser::Param() {
  auto type = Type();
  auto id = Identifier();
  if (!type || !id) {
    return llvm::None;
  }

  return source::ParamDecl(*id, source::NamedType(*type));
}

/// Block = "{" "}";
llvm::Optional<LexicalScope> Parser::Block() {
  auto lKeyBrace = Match({TK_LKeyBracket});
  auto rKeyBrace = Match({TK_RKeyBracket});
  if (!lKeyBrace || !rKeyBrace) {
    return llvm::None;
  }

  return LexicalScope{*lKeyBrace, *rKeyBrace};
}

/// Function_type = "proc" | "fn";
llvm::Optional<Token> Parser::FunctionType() {
  return Match({TK_ProcType, TK_FnType});
}

/// Type = Identifier;
llvm::Optional<source::NamedType> Parser::Type() {
  auto id = Identifier();
  if (!id) {
    return llvm::None;
  }

  return source::NamedType(*id);
}

/// Identifier = [a-Z]*;
llvm::Optional<Token> Parser::Identifier() { return Match({TK_Identifier}); }

llvm::Optional<Token> Parser::Match(llvm::ArrayRef<TokenKind> list) {
  Token tk = Lex.prev();
  if (!llvm::is_contained(list, tk.Kind)) {
    Report(tk, list, llvm::errs());
    return llvm::None;
  }

  Lex.next();
  return tk;
}

llvm::Optional<Token> Parser::TryMatch(llvm::ArrayRef<TokenKind> list,
                                       bool consumeTokenIfMatch) {
  Token tk = Lex.prev();
  if (!llvm::is_contained(list, tk.Kind)) {
    return llvm::None;
  }

  if (consumeTokenIfMatch) {
    Lex.next();
  }

  return tk;
}

llvm::Optional<source::CompilationUnit> tmplang::Parse(tmplang::Lexer &lex) {
  return Parser(lex).Start();
}
