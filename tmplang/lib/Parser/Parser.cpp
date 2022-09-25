#include "tmplang/AST/Decls.h"
#include "tmplang/AST/Types.h"
#include "tmplang/Lexer/Lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <tmplang/AST/ASTContext.h>
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
  Parser(Lexer &lex, ASTContext &ctx) : Ctx(ctx), Lex(lex) {
    // Retrieve first token
    Lex.next();
  }

  llvm::Optional<CompilationUnit> Start();

private:
  llvm::Optional<FunctionDecl> FunctionDefinition();
  llvm::Optional<std::vector<ParamDecl>> ParamList();
  llvm::Optional<ParamDecl> Param();
  bool Block();
  llvm::Optional<FunctionDecl::SubroutineKind> FunctionType();
  NamedType *Type();
  llvm::Optional<llvm::StringRef> Identifier();

private:
  bool Match(llvm::ArrayRef<TokenKind> list);
  bool TryMatch(llvm::ArrayRef<TokenKind> list,
                bool consumeTokenIfMatch = false);

  ASTContext &Ctx;
  Lexer &Lex;
};

} // namespace

/// Start = Function_Definition*;
///       | EOF;
llvm::Optional<CompilationUnit> Parser::Start() {
  CompilationUnit result;
  while (true) {
    if (TryMatch({TokenKind::TK_EOF}, /*consumeTok*/ true)) {
      return result;
    }

    llvm::Optional<FunctionDecl> functionDef = FunctionDefinition();
    if (!functionDef) {
      return llvm::None;
    }
    result.AddFunctionDecl(std::move(*functionDef));
  }
  return result;
}

/// Function_Definition =
///  [1] | Function_Type, Identifier, ":", Param_List, "->", Type, Block
///  [2] | Function_Type, Identifier, ":", Param_List, Block
///  [3] | Function_Type, Identifier, "->", Type, Block
///  [4] | Function_Type, Identifier, Block;
llvm::Optional<FunctionDecl> Parser::FunctionDefinition() {
  llvm::Optional<FunctionDecl::SubroutineKind> subroutineKind = FunctionType();
  if (!subroutineKind) {
    return llvm::None;
  }
  llvm::Optional<llvm::StringRef> functionId = Identifier();
  if (!functionId) {
    return llvm::None;
  }
  // [1] && [2]
  if (TryMatch({TK_Colon}, /*consumeTok*/ true)) {
    llvm::Optional<std::vector<ParamDecl>> params = ParamList();
    if (!params) {
      return llvm::None;
    }

    if (TryMatch({TK_RArrow}, /*consumeTok*/ true)) {
      NamedType *returnTy = Type();
      if (!returnTy || !Block()) {
        return llvm::None;
      }
      // [1]
      return FunctionDecl(*subroutineKind, *functionId, *returnTy, *params);
    }

    // [2]
    if (!Block()) {
      return llvm::None;
    }
    return FunctionDecl(*subroutineKind, *functionId,
                        BuiltinType::getType(Ctx, BuiltinType::K_Unit),
                        *params);
  }

  if (TryMatch({TK_RArrow}, /*consumeTok*/ true)) {
    NamedType *returnTy = Type();
    if (!returnTy || !Block()) {
      return llvm::None;
    }
    // [3]
    return FunctionDecl(*subroutineKind, *functionId, *returnTy, {});
  }

  if (!Block()) {
    return llvm::None;
  }
  // [4]
  return FunctionDecl(*subroutineKind, *functionId,
                      BuiltinType::getType(Ctx, BuiltinType::K_Unit), {});
}

/// Param_List = Param (",", Param)*;
llvm::Optional<std::vector<ParamDecl>> Parser::ParamList() {
  std::vector<ParamDecl> result;
  llvm::Optional<ParamDecl> par = Param();
  if (!par) {
    return llvm::None;
  }
  result.push_back(std::move(*par));

  while (TryMatch({TK_Comma}, /*consumeTok*/ true)) {
    llvm::Optional<ParamDecl> par = Param();
    if (!par) {
      return llvm::None;
    }
    result.push_back(std::move(*par));
  }

  return result;
}

/// Param = Type Identifier;
llvm::Optional<ParamDecl> Parser::Param() {
  NamedType *ty = Type();
  if (!ty) {
    return llvm::None;
  }
  llvm::Optional<llvm::StringRef> id = Identifier();
  if (!id) {
    return llvm::None;
  }
  return ParamDecl(*id, *ty);
}

/// Block = "{" "}";
bool Parser::Block() {
  return Match({TK_LKeyBracket}) && Match({TK_RKeyBracket});
}

/// Function_type = "proc" | "fn";
llvm::Optional<FunctionDecl::SubroutineKind> Parser::FunctionType() {
  TokenKind tk = Lex.prev().Kind;
  if (!Match({TK_ProcType, TK_FnType})) {
    return llvm::None;
  }
  switch (tk) {
  case TK_FnType:
    return FunctionDecl::Function;
  case TK_ProcType:
    return FunctionDecl::Procedure;
  default:
    llvm_unreachable("Only function or procedure tokens can get here");
  }
}

/// Type = Identifier;
NamedType *Parser::Type() {
  llvm::Optional<llvm::StringRef> typeName = Identifier();
  if (!typeName) {
    return nullptr;
  }
  return &Ctx.getNamedType(*typeName);
}

/// Identifier = [a-Z]*;
llvm::Optional<llvm::StringRef> Parser::Identifier() {
  if (!Match({TK_Identifier})) {
    return llvm::None;
  };
  // I need source information here :(
  return llvm::StringRef("id");
}

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

llvm::Optional<CompilationUnit> tmplang::Parse(tmplang::Lexer &lex,
                                               ASTContext &ctx) {
  return Parser(lex, ctx).Start();
}
