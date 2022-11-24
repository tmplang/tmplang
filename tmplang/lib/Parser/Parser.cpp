#include <tmplang/Parser/Parser.h>

#include <tmplang/Diagnostics/Diagnostic.h>
#include <tmplang/Diagnostics/Hint.h>
#include <tmplang/Tree/Source/Decls.h>
#include <tmplang/Tree/Source/Types.h>

using namespace tmplang;

namespace {

struct LexicalScope {
  Token LKeyBracket;
  Token RKeyBracket;
};

class Parser {
public:
  Parser(Lexer &lex, llvm::raw_ostream &out, const SourceManager &sm)
      : Lex(lex), Out(out), SM(sm) {}

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

  // Utility functions
  Token getPrevToken() const;
  Token getCurrToken() const;
  Token peakNextToken() const;

  SourceLocation getStartCurrToken() const;
  SourceLocation getEndCurrToken() const;

  bool emitUnknownToken(const bool force = false) const;

private:
  /// Checks if the current tokens is any of the \ref expected tokens. If so
  /// returns current token
  llvm::Optional<Token> is(llvm::ArrayRef<TokenKind> expected) const;

  /// Checks if the current tokens is any of the \ref expected tokens. If so
  /// returns current token and advances to the next token
  llvm::Optional<Token> consume(llvm::ArrayRef<TokenKind> expected);

  Lexer &Lex;
  llvm::raw_ostream &Out;
  const SourceManager &SM;
};

} // namespace

/// Start = Function_Definition*;
///       | EOF;
llvm::Optional<source::CompilationUnit> Parser::Start() {
  source::CompilationUnit compilationUnit;

  while (true) {
    if (consume({TokenKind::TK_EOF})) {
      return compilationUnit;
    }

    auto func = FunctionDefinition();
    if (!func) {
      // Nothing to do, already reported
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
    if (getCurrToken().Kind == TK_Identifier) {
      Diagnostic(DiagId::err_missing_subprogram_class,
                 getCurrToken().SrcLocSpan,
                 PreprendHint(getCurrToken().SrcLocSpan.Start,
                              {ToString(TK_FnType), ToString(TK_ProcType)}))
          .print(Out, SM);
    } else {
      emitUnknownToken(/*force*/ true);
    }
    return llvm::None;
  }

  auto id = Identifier();
  if (!id) {
    Diagnostic(DiagId::err_missing_subprogram_id, getCurrToken().SrcLocSpan,
               InsertTextAtHint(getStartCurrToken(), "<subprogram_identifier>"))
        .print(Out, SM);
    return llvm::None;
  }

  // [1] && [2]
  if (auto colon = consume({TK_Colon})) {
    auto paramList = ParamList();
    if (!paramList) {
      // Nothing to report here, reported on ParamList
      return llvm::None;
    }

    if (auto arrow = consume({TK_RArrow})) {
      // [1]
      auto returnType = Type();
      if (!returnType) {
        // Nothing to report here, reported on Type
        return llvm::None;
      }

      auto block = Block();
      if (!block) {
        // Nothing to report here, reported on Block
        return llvm::None;
      }

      return source::FunctionDecl::Create(
          *funcType, *id, *colon, std::move(*paramList),
          source::FunctionDecl::ArrowAndType{*arrow, std::move(returnType)},
          block->LKeyBracket, block->RKeyBracket);
    }

    constexpr TokenKind firstTokensOfType[] = {TK_Identifier, TK_LParentheses};
    if (is(firstTokensOfType)) {
      // Missing arrow
      Diagnostic(DiagId::err_missing_arrow, getPrevToken().SrcLocSpan,
                 InsertTextAtHint(getPrevToken().SrcLocSpan.End + 1,
                                  ToString(TK_RArrow)))
          .print(Out, SM);
      return llvm::None;
    }

    // [2]
    auto block = Block();
    if (!block) {
      // Nothing to report here, reported on Block
      return llvm::None;
    }

    return source::FunctionDecl::Create(*funcType, *id, *colon,
                                        std::move(*paramList),
                                        block->LKeyBracket, block->RKeyBracket);
  }

  if (auto arrow = consume({TK_RArrow})) {
    // [3]
    auto returnType = Type();
    if (!returnType) {
      // Nothing to report here, reported on Type
      return llvm::None;
    }

    auto block = Block();
    if (!block) {
      // Nothing to report here, reported on Block
      return llvm::None;
    }

    return source::FunctionDecl::Create(
        *funcType, *id,
        source::FunctionDecl::ArrowAndType{*arrow, std::move(returnType)},
        block->LKeyBracket, block->RKeyBracket);
  }

  constexpr TokenKind firstTokensOfType[] = {TK_Identifier, TK_LParentheses};
  if (is(firstTokensOfType)) {
    // Missing arrow
    Diagnostic(DiagId::err_missing_arrow, getPrevToken().SrcLocSpan,
               InsertTextAtHint(getPrevToken().SrcLocSpan.End + 1,
                                ToString(TK_RArrow)))
        .print(Out, SM);
    return llvm::None;
  }

  if (emitUnknownToken()) {
    return llvm::None;
  }

  // [4]
  auto block = Block();
  if (!block) {
    // Nothing to report here, reported on Block
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
    // Nothing to report here, reported on Param
    return llvm::None;
  }

  paramList.Elems.push_back(std::move(*firstParam));

  while (auto comma = consume({TK_Comma})) {
    auto param = Param();
    if (!param) {
      // Nothing to report here, reported on Param
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
    // Nothing to report here, reported on Type
    return llvm::None;
  }

  auto id = Identifier();
  if (!id) {
    Diagnostic(DiagId::err_missing_variable_identifier_after_type,
               getCurrToken().SrcLocSpan,
               InsertTextAtHint(getStartCurrToken(), "<parameter_id>"))
        .print(Out, SM);
    return llvm::None;
  }

  return source::ParamDecl(std::move(type), *id);
}

/// Block = "{" "}";
llvm::Optional<LexicalScope> Parser::Block() {
  if (emitUnknownToken()) {
    return llvm::None;
  }

  auto lKeyBrace = consume({TK_LKeyBracket});
  if (!lKeyBrace) {
    Diagnostic(DiagId::err_missing_left_key_brace, getPrevToken().SrcLocSpan,
               InsertTextAtHint(getPrevToken().SrcLocSpan.End + 1, "{"))
        .print(Out, SM);
    return llvm::None;
  }

  if (emitUnknownToken()) {
    return llvm::None;
  }

  auto rKeyBrace = consume({TK_RKeyBracket});
  if (!rKeyBrace) {
    Diagnostic(DiagId::err_missing_right_key_brace, getPrevToken().SrcLocSpan,
               InsertTextAtHint(getPrevToken().SrcLocSpan.End + 1, "}"))
        .print(Out, SM);
    return llvm::None;
  }

  return LexicalScope{*lKeyBrace, *rKeyBrace};
}

/// Function_type = "proc" | "fn";
llvm::Optional<Token> Parser::FunctionType() {
  return consume({TK_ProcType, TK_FnType});
}

/// Type = NamedType | TupleType;
source::RAIIType Parser::Type() {
  constexpr TokenKind firstTokensOfNamedType[] = {TK_Identifier};
  if (is(firstTokensOfNamedType)) {
    return NamedType();
  }

  constexpr TokenKind firstTokensOfTupleType[] = {TK_LParentheses};
  if (is(firstTokensOfTupleType)) {
    return TupleType();
  }

  const TokenKind currTokenKind = getCurrToken().Kind;

  if (emitUnknownToken()) {
    return nullptr;
  }

  if (currTokenKind == TK_RParentheses && getPrevToken().Kind != TK_Comma) {
    Diagnostic(DiagId::err_missing_left_parenthesis_opening_tuple,
               getCurrToken().SrcLocSpan,
               InsertTextAtHint(getPrevToken().SrcLocSpan.End + 1, "("))
        .print(Out, SM);
  } else {
    Diagnostic(
        DiagId::err_missing_type, getPrevToken().SrcLocSpan,
        InsertTextAtHint(getPrevToken().SrcLocSpan.End + 1, "<type>", " ", " "))
        .print(Out, SM);
  }

  return nullptr;
}

/// NamedType = Identifier;
source::RAIIType Parser::NamedType() {
  auto id = Identifier();
  assert(id && "This is validated on the call-site");

  return source::make_RAIIType<source::NamedType>(*id);
}

/// TupleType = "(" ( Type ("," Type)* )? ")";
source::RAIIType Parser::TupleType() {
  auto lparentheses = consume({TK_LParentheses});
  assert(lparentheses && "This is validated on the call-site");

  llvm::SmallVector<source::RAIIType, 4> types;
  llvm::SmallVector<Token, 3> commas;

  constexpr TokenKind firstTokensOfType[] = {TK_LParentheses, TK_Identifier};
  if (is(firstTokensOfType)) {
    auto firstType = Type();
    if (!firstType) {
      // Nothing to report here, reported on Type
      return nullptr;
    }

    types.push_back(std::move(firstType));

    while (auto comma = consume({TK_Comma})) {
      auto followingType = Type();
      if (!followingType) {
        // Nothing to do here, reported on Type
        return nullptr;
      }

      types.push_back(std::move(followingType));
      commas.push_back(std::move(*comma));
    }
  }

  auto rparentheses = consume({TK_RParentheses});
  if (!rparentheses) {
    // In these cases:
    //   [1] fn foo: (i32 i32 )   [...]
    //   [2] fn foo: (i32 i32 var [...]
    //   [3] fn foo: (i32 i32 ,   [...]
    //   comma here -----^
    // else:
    //   [eg.] fn foo: (i32 var { [...]
    //     r-par here -----^

    DiagId id = DiagId::err_missing_right_parenthesis_closing_tuple;
    llvm::StringRef toInsert = ")";

    if (getCurrToken().Kind == TK_Identifier &&
        (peakNextToken().Kind == TK_RParentheses ||
         peakNextToken().Kind == TK_Identifier ||
         peakNextToken().Kind == TK_Comma)) {
      id = DiagId::err_missing_comma_prior_tuple_param;
      toInsert = ",";
    }

    Diagnostic(id, getCurrToken().SrcLocSpan,
               InsertTextAtHint(getStartCurrToken(), toInsert))
        .print(Out, SM);
    return nullptr;
  }

  return source::make_RAIIType<source::TupleType>(
      *lparentheses, std::move(types), std::move(commas), *rparentheses);
}

/// Identifier = [a-zA-Z][a-zA-Z0-9]*;
llvm::Optional<Token> Parser::Identifier() { return consume({TK_Identifier}); }

Token Parser::getPrevToken() const { return Lex.getPrevToken(); }
Token Parser::getCurrToken() const { return Lex.getCurrentToken(); }
Token Parser::peakNextToken() const { return Lex.peakNextToken(); }

SourceLocation Parser::getStartCurrToken() const {
  return getCurrToken().SrcLocSpan.Start;
}
SourceLocation Parser::getEndCurrToken() const {
  return getCurrToken().SrcLocSpan.End;
}

bool Parser::emitUnknownToken(const bool force) const {
  if (force || is(TK_Unknown)) {
    Diagnostic(DiagId::err_found_unknown_token, getCurrToken().SrcLocSpan,
               NoHint())
        .print(Out, SM);
    return true;
  }
  return false;
}

llvm::Optional<Token> Parser::is(llvm::ArrayRef<TokenKind> expected) const {
  const Token tk = Lex.getCurrentToken();
  return llvm::is_contained(expected, tk.Kind) ? tk : llvm::Optional<Token>{};
}

llvm::Optional<Token> Parser::consume(llvm::ArrayRef<TokenKind> expected) {
  if (auto tk = is(expected)) {
    Lex.next();
    return tk;
  }

  return llvm::None;
}

llvm::Optional<source::CompilationUnit>
tmplang::Parse(tmplang::Lexer &lex, llvm::raw_ostream &out,
               const SourceManager &sm) {
  return Parser(lex, out, sm).Start();
}
