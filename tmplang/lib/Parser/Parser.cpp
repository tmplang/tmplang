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
  Parser(Lexer &lex) : Lex(lex) {}

  llvm::Optional<source::CompilationUnit> Start();

private:
  llvm::Optional<source::FunctionDecl> FunctionDefinition();
  llvm::Optional<source::CommaSeparatedList<source::ParamDecl, 4>> ParamList();
  llvm::Optional<source::ParamDecl> Param();
  llvm::Optional<LexicalScope> Block();
  llvm::Optional<Token> FunctionType();
  llvm::Optional<Token> Identifier();

  // Types...
  source::RAIIType Type();
  source::RAIIType NamedType();
  source::RAIIType TupleType();

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
  if (!funcType) {
    return llvm::None;
  }

  auto id = Identifier();
  if (!id) {
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
      if (!returnType) {
        return llvm::None;
      }

      auto block = Block();
      if (!block) {
        return llvm::None;
      }

      return source::FunctionDecl::Create(
          *funcType, *id, *colon, std::move(*paramList),
          source::FunctionDecl::ArrowAndType{*arrow, std::move(returnType)},
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
    if (!returnType) {
      return llvm::None;
    }

    auto block = Block();
    if (!block) {
      return llvm::None;
    }

    return source::FunctionDecl::Create(
        *funcType, *id,
        source::FunctionDecl::ArrowAndType{*arrow, std::move(returnType)},
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
llvm::Optional<source::CommaSeparatedList<source::ParamDecl, 4>>
Parser::ParamList() {
  source::CommaSeparatedList<source::ParamDecl, 4> paramList;

  auto firstParam = Param();
  if (!firstParam) {
    return llvm::None;
  }

  paramList.Elems.push_back(std::move(*firstParam));

  while (auto comma = TryMatch({TK_Comma}, /*consumeTok*/ true)) {
    auto param = Param();
    if (!param) {
      return llvm::None;
    }

    paramList.Elems.push_back(std::move(*param));
    paramList.Commas.push_back(*comma);
  }

  return paramList;
}

/// Param = Type Identifier;
llvm::Optional<source::ParamDecl> Parser::Param() {
  auto type = Type();
  if (!type) {
    return llvm::None;
  }

  auto id = Identifier();
  if (!id) {
    return llvm::None;
  }

  return source::ParamDecl(std::move(type), *id);
}

/// Block = "{" "}";
llvm::Optional<LexicalScope> Parser::Block() {
  auto lKeyBrace = Match({TK_LKeyBracket});
  if (!lKeyBrace) {
    return llvm::None;
  }

  auto rKeyBrace = Match({TK_RKeyBracket});
  if (!rKeyBrace) {
    return llvm::None;
  }

  return LexicalScope{*lKeyBrace, *rKeyBrace};
}

/// Function_type = "proc" | "fn";
llvm::Optional<Token> Parser::FunctionType() {
  return Match({TK_ProcType, TK_FnType});
}

/// Type = NamedType | TupleType;
source::RAIIType Parser::Type() {
  constexpr TokenKind firstTokensOfNamedType[] = {TK_Identifier};
  if (TryMatch(firstTokensOfNamedType, /*consumeTokenIfMatch=*/false)) {
    return NamedType();
  }

  constexpr TokenKind firstTokensOfTupleType[] = {TK_LParentheses};
  if (TryMatch(firstTokensOfTupleType, /*consumeTokenIfMatch=*/false)) {
    return TupleType();
  }

  return nullptr;
}

/// NamedType = Identifier;
source::RAIIType Parser::NamedType() {
  auto id = Identifier();
  if (!id) {
    return nullptr;
  }

  return source::make_RAIIType<source::NamedType>(*id);
}

/// TupleType = "(" ( Type ("," Type)* )? ")";
source::RAIIType Parser::TupleType() {
  auto lparentheses = Match({TK_LParentheses});
  if (!lparentheses) {
    return nullptr;
  }

  llvm::SmallVector<source::RAIIType, 4> types;
  llvm::SmallVector<Token, 3> commas;

  constexpr TokenKind firstTokensOfType[] = {TK_LParentheses, TK_Identifier};
  if (TryMatch(firstTokensOfType, /*consumeTokenIfMatch=*/false)) {
    auto firstType = Type();
    if (!firstType) {
      // TODO: Emit error
      return nullptr;
    }

    types.push_back(std::move(firstType));

    while (auto comma = TryMatch({TK_Comma}, /*consumeTokenIfMatch=*/true)) {
      auto followingType = Type();
      if (!followingType) {
        // TODO: Emit error
        return nullptr;
      }

      types.push_back(std::move(followingType));
      commas.push_back(std::move(*comma));
    }
  }

  auto rparentheses = Match({TK_RParentheses});
  if (!rparentheses) {
    return nullptr;
  }

  return source::make_RAIIType<source::TupleType>(
      *lparentheses, std::move(types), std::move(commas), *rparentheses);
}

/// Identifier = [a-zA-Z][a-zA-Z0-9]*;
llvm::Optional<Token> Parser::Identifier() { return Match({TK_Identifier}); }

llvm::Optional<Token> Parser::Match(llvm::ArrayRef<TokenKind> list) {
  Token tk = Lex.getCurrentToken();
  if (!llvm::is_contained(list, tk.Kind)) {
    Report(tk, list, llvm::errs());
    return llvm::None;
  }

  Lex.next();
  return tk;
}

llvm::Optional<Token> Parser::TryMatch(llvm::ArrayRef<TokenKind> list,
                                       bool consumeTokenIfMatch) {
  Token tk = Lex.getCurrentToken();
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
