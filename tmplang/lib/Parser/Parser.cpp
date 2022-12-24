#include <tmplang/Parser/Parser.h>

#include <tmplang/Diagnostics/Diagnostic.h>
#include <tmplang/Diagnostics/Hint.h>
#include <tmplang/Tree/Source/Decls.h>
#include <tmplang/Tree/Source/Exprs.h>
#include <tmplang/Tree/Source/Types.h>

using namespace tmplang;

namespace {

struct LexicalScope {
  Token LKeyBracket;
  Token RKeyBracket;
};

using RAIIExpr = std::unique_ptr<source::Expr>;

class Parser {
public:
  Parser(Lexer &lex, raw_ostream &out, const SourceManager &sm)
      : Lex(lex), Out(out), SM(sm) {
    // These two "consume" calls will initialize the current and next token
    consume();
    consume();
  }

  Optional<source::CompilationUnit> Start();

private:
  Optional<source::FunctionDecl> FunctionDefinition();
  Optional<source::FunctionDecl> ArrowAndEndOfSubprogramFactored(
      Token funcType, Token id, Optional<Token> colon = None,
      SmallVector<source::ParamDecl, 4> paramList = {});
  Optional<Token> missingSubprogramTypeRecovery();
  Optional<Token> missingSubprogramIdRecovery();
  Optional<Token> missingReturnTypeArrowRecovery();

  Optional<SmallVector<source::ParamDecl, 4>> ParamList();
  Optional<Token> missingCommaParamSepRecovery();

  Optional<source::ParamDecl> Param();
  Optional<Token> missingVariableOnParamRecovery();

  Optional<source::FunctionDecl::Block> Block();
  Optional<Token> missingLeftKeyBracketRecovery();
  Optional<Token> missingRightKeyBracketRecovery();

  Optional<std::vector<source::ExprStmt>> ExprList();

  RAIIExpr Expr();
  Optional<Token> missingSemicolonAfterExpressionRecovery();

  Optional<Token> Identifier();
  Optional<Token> Number();

  // Types...
  source::RAIIType Type();
  source::RAIIType NamedType();
  source::RAIIType TupleType();
  Optional<Token> missingCommaBetweenTypesRecovery();
  Optional<Token> missingRightParenthesesOfTupleRecovery();

  // Simple token parsing. This function is useful in conjuntion with
  // parseOrRecover so it can recieve the address
  template <TokenKind... Kinds> Optional<Token> ParseToken() {
    if (tk().isOneOf(Kinds...)) {
      return consume();
    }
    return None;
  }

  // Utility functions
  const Token &prevTk() const;
  const Token &tk() const;
  const Token &nextTk() const;

  SourceLocation getStartCurrToken() const;
  SourceLocation getEndCurrToken() const;

  bool emitUnknownTokenDiag(const bool force = false) const;

  // Returns current token and retrives next one, updating prevTk, tk and
  // nextTk
  Token consume();

  // Utility functions to build recovery Tokens, and error reporting
  Token getRecoveryToken(TokenKind kind);
  Token getRecoveryToken(StringRef id, TokenKind kind);

  template <Optional<Token> (Parser::*parsingFunc)(),
            Optional<Token> (Parser::*recoveryFunc)() = nullptr>
  Optional<Token> parseOrTryRecover(bool emitUnexpectedTokenDiag = true) {
    // Try the normal path, parse as it is expected
    if (auto parsedToken = (this->*parsingFunc)()) {
      return *parsedToken;
    }

    // If there is no recovery function, do not recover
    if (recoveryFunc == nullptr) {
      return None;
    }

    // Try to perform the actual recovery
    if (auto recoveryToken = (this->*recoveryFunc)()) {
      return *recoveryToken;
    }

    // If asked, report a default unkown token message
    if (emitUnexpectedTokenDiag) {
      // FIXME: This should be: found X when Y was expected
      Diagnostic(DiagId::err_found_unknown_token, tk().getSpan(), NoHint())
          .print(Out, SM);
    }

    return None;
  }

private:
  Lexer &Lex;
  raw_ostream &Out;
  const SourceManager &SM;

  struct {
    Token PrevToken;
    Token CurrToken;
    Token NextToken;
    unsigned NumberOfRecoveriesPerformed = 0;
  } ParserState;
};

} // namespace

/// Start = Function_Definition*;
///       | EOF;
Optional<source::CompilationUnit> Parser::Start() {
  std::vector<source::FunctionDecl> functionDeclarations;
  while (true) {
    if (tk().is(TokenKind::TK_EOF)) {
      return source::CompilationUnit(std::move(functionDeclarations),
                                     ParserState.NumberOfRecoveriesPerformed);
    }

    auto func = FunctionDefinition();
    if (!func) {
      // Nothing to do, already reported
      return None;
    }

    functionDeclarations.push_back(std::move(*func));
  }

  return source::CompilationUnit(std::move(functionDeclarations),
                                 ParserState.NumberOfRecoveriesPerformed);
}

Optional<source::FunctionDecl> Parser::ArrowAndEndOfSubprogramFactored(
    Token funcType, Token id, Optional<Token> colon,
    SmallVector<source::ParamDecl, 4> paramList) {
  if (auto arrow = parseOrTryRecover<&Parser::ParseToken<TK_RArrow>,
                                     &Parser::missingReturnTypeArrowRecovery>(
          /*emitUnexpectedTokenDiag*/ false)) {
    // [1]
    auto returnType = Type();
    if (!returnType) {
      // Nothing to report here, reported on Type
      return None;
    }

    auto block = Block();
    if (!block) {
      // Nothing to report here, reported on Block
      return None;
    }

    return source::FunctionDecl(
        funcType, id, std::move(*block), colon, std::move(paramList),
        source::FunctionDecl::ArrowAndType{*arrow, std::move(returnType)});
  }

  // [2]
  auto block = Block();
  if (!block) {
    // Nothing to report here, reported on Block
    return None;
  }

  return source::FunctionDecl(funcType, id, std::move(*block), colon,
                              std::move(paramList));
}

/// Function_type = "proc" | "fn";

/// Function_Definition =
///  [1] | Function_Type, Identifier, ":", Param_List, "->", Type, Block
///  [2] | Function_Type, Identifier, ":", Param_List, Block
///  [3] | Function_Type, Identifier, "->", Type, Block
///  [4] | Function_Type, Identifier, Block;
Optional<source::FunctionDecl> Parser::FunctionDefinition() {
  auto funcType = parseOrTryRecover<&Parser::ParseToken<TK_FnType, TK_ProcType>,
                                    &Parser::missingSubprogramTypeRecovery>();
  if (!funcType) {
    // Since this is the start of top level declaration, lets consume the
    // unknowns until we find something we understand
    if (tk().is(tmplang::TK_Unknown)) {
      consume();
      return FunctionDefinition();
    }
    return None;
  }

  auto id = parseOrTryRecover<&Parser::Identifier,
                              &Parser::missingSubprogramIdRecovery>();
  if (!id) {
    return None;
  }

  // [1] && [2]
  if (tk().is(TK_Colon)) {
    auto colon = consume();

    auto paramList = ParamList();
    if (!paramList) {
      // Nothing to report here, reported on ParamList
      return None;
    }

    return ArrowAndEndOfSubprogramFactored(*funcType, *id, colon,
                                           std::move(*paramList));
  }

  return ArrowAndEndOfSubprogramFactored(*funcType, *id);
}

Optional<Token> Parser::missingSubprogramTypeRecovery() {
  const bool potentialStartOfSubprogram =
      tk().is(TK_Identifier) &&
      nextTk().isOneOf(TK_Colon, TK_RArrow, TK_LKeyBracket);
  if (!potentialStartOfSubprogram) {
    return None;
  }

  Diagnostic(DiagId::err_missing_subprogram_class, tk().getSpan(),
             InsertTextAtHint(tk().getSpan().Start,
                              {ToString(TK_FnType), ToString(TK_ProcType)}, ""))
      .print(Out, SM);

  return getRecoveryToken(TK_FnType);
}

Optional<Token> Parser::missingSubprogramIdRecovery() {
  const bool missingId = prevTk().isOneOf(TK_FnType, TK_ProcType) &&
                         tk().isOneOf(TK_Colon, TK_RArrow, TK_LKeyBracket);
  if (!missingId) {
    return None;
  }

  constexpr StringLiteral placeHolder = "<subprogram_identifier>";

  Diagnostic(DiagId::err_missing_subprogram_id, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), placeHolder))
      .print(Out, SM);

  return getRecoveryToken(placeHolder, TK_Identifier);
}

Optional<Token> Parser::missingReturnTypeArrowRecovery() {
  const bool missingArrow =
      tk().isOneOf(/*firstTokensOfType*/ TK_Identifier, TK_LParentheses);
  if (!missingArrow) {
    return None;
  }

  Diagnostic(DiagId::err_missing_arrow, prevTk().getSpan(),
             InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_RArrow)))
      .print(Out, SM);

  return getRecoveryToken(TK_RArrow);
}

/// Param_List = Param (",", Param)*;
Optional<SmallVector<source::ParamDecl, 4>> Parser::ParamList() {
  SmallVector<source::ParamDecl, 4> paramList;

  auto firstParam = Param();
  if (!firstParam) {
    // Nothing to report here, reported on Param
    return None;
  }
  paramList.push_back(std::move(*firstParam));

  while (auto comma = parseOrTryRecover<&Parser::ParseToken<TK_Comma>,
                                        &Parser::missingCommaParamSepRecovery>(
             /*emitUnexpectedTokenDiag*/ false)) {
    paramList.back().setComma(*comma);

    auto param = Param();
    if (!param) {
      // Nothing to report here, reported on Param
      return None;
    }

    paramList.push_back(std::move(*param));
  }

  return paramList;
}

Optional<Token> Parser::missingCommaParamSepRecovery() {
  // fn foo: i32 var i32 var ...
  //                ^___ comma here

  // fn foo: i32 var () var ...
  // fn foo: i32 var (var) var ...
  //   do not emit in the above cases, the can get confusing with missing arrow
  const bool missingComma = prevTk().is(TK_Identifier) &&
                            tk().is(TK_Identifier) &&
                            nextTk().is(TK_Identifier);
  if (!missingComma) {
    return None;
  }

  Diagnostic(DiagId::err_missing_comma, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  return getRecoveryToken(TK_Comma);
}

/// Param = Type Identifier;
Optional<source::ParamDecl> Parser::Param() {
  auto type = Type();
  if (!type) {
    // Nothing to report here, reported on Type
    return None;
  }

  auto paramId = parseOrTryRecover<&Parser::Identifier,
                                   &Parser::missingVariableOnParamRecovery>();
  if (!paramId) {
    return None;
  }

  return source::ParamDecl(std::move(type), *paramId);
}

Optional<Token> Parser::missingVariableOnParamRecovery() {
  constexpr StringLiteral paramId = "<parameter_id>";

  Diagnostic(DiagId::err_missing_variable_identifier_after_type, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), paramId))
      .print(Out, SM);

  return getRecoveryToken(paramId, TK_Identifier);
}

/// Block = "{" ExprList "}";
Optional<source::FunctionDecl::Block> Parser::Block() {
  auto lKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_LKeyBracket>,
                        &Parser::missingLeftKeyBracketRecovery>();
  if (!lKeyBracket) {
    return None;
  }

  Optional<std::vector<source::ExprStmt>> exprs = ExprList();
  if (!exprs) {
    // Errors already reported
    return None;
  }

  auto rKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_RKeyBracket>,
                        &Parser::missingRightKeyBracketRecovery>();
  if (!rKeyBracket) {
    return None;
  }

  return source::FunctionDecl::Block{*lKeyBracket, std::move(*exprs),
                                     *rKeyBracket};
}

Optional<Token> Parser::missingLeftKeyBracketRecovery() {
  Diagnostic(
      DiagId::err_missing_left_key_brace, prevTk().getSpan(),
      InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_LKeyBracket)))
      .print(Out, SM);

  return getRecoveryToken(TK_LKeyBracket);
}

Optional<Token> Parser::missingRightKeyBracketRecovery() {
  Diagnostic(
      DiagId::err_missing_right_key_brace, prevTk().getSpan(),
      InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_RKeyBracket)))
      .print(Out, SM);

  return getRecoveryToken(TK_RKeyBracket);
}

/// Type = NamedType | TupleType;
source::RAIIType Parser::Type() {
  if (tk().is(/*firstTokensOfNamedType*/ TK_Identifier)) {
    return NamedType();
  }

  if (tk().is(/*firstTokensOfTupleType*/ TK_LParentheses)) {
    return TupleType();
  }

  if ((prevTk().isOneOf(TK_Colon, TK_RArrow) && tk().is(TK_RParentheses)) ||
      (prevTk().is(TK_Comma) && tk().is(TK_RParentheses) &&
       nextTk().isOneOf(TK_RParentheses, TK_Comma))) {
    Diagnostic(
        DiagId::err_missing_left_parenthesis_opening_tuple, tk().getSpan(),
        InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_LParentheses)))
        .print(Out, SM);
  } else if ((prevTk().is(TK_RArrow) && tk().is(TK_LKeyBracket)) ||
             (prevTk().is(TK_Comma) && tk().is(TK_RParentheses))) {
    Diagnostic(DiagId::err_missing_type, prevTk().getSpan(),
               InsertTextAtHint(prevTk().getSpan().End + 1, "<type>"))
        .print(Out, SM);
  } else {
    emitUnknownTokenDiag();
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
  Token lparentheses = consume();
  assert(lparentheses.is(TK_LParentheses) &&
         "This is validated on the call-site");

  SmallVector<source::RAIIType, 4> types;
  SmallVector<Token, 3> commas;

  if (tk().isOneOf(/*firstTokensOfType*/ TK_LParentheses, TK_Identifier)) {
    auto firstType = Type();
    if (!firstType) {
      // Nothing to report here, reported on Type
      return nullptr;
    }

    types.push_back(std::move(firstType));

    while (auto comma =
               parseOrTryRecover<&Parser::ParseToken<TK_Comma>,
                                 &Parser::missingCommaBetweenTypesRecovery>(
                   /*emitUnexpectedTokenDiag*/ false)) {
      auto followingType = Type();
      if (!followingType) {
        // Nothing to do here, reported on Type
        return nullptr;
      }

      types.push_back(std::move(followingType));
      commas.push_back(std::move(*comma));
    }
  }

  auto rparentheses =
      parseOrTryRecover<&Parser::ParseToken<TK_RParentheses>,
                        &Parser::missingRightParenthesesOfTupleRecovery>();
  assert(rparentheses && "This is unconditionally valid");

  return source::make_RAIIType<source::TupleType>(
      lparentheses, std::move(types), std::move(commas),
      std::move(*rparentheses));
}

Optional<Token> Parser::missingCommaBetweenTypesRecovery() {
  // fn foo: ( ()  i32) var
  // fn foo: (i32  i32) var
  // fn foo: (i32  ()) var
  // fn foo: (i32  i32) var
  //             ^___ comma here

  // But not in the following cases:
  //   fn foo: (i32  var {}
  const bool missingComma =
      prevTk().isOneOf(TK_Identifier, TK_RParentheses) &&
      tk().isOneOf(TK_Identifier, TK_LParentheses) &&
      (nextTk().isNot(TK_Identifier) && nextTk().isNot(TK_LKeyBracket));
  if (!missingComma) {
    return None;
  }

  Diagnostic(DiagId::err_missing_comma_prior_tuple_param, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  return getRecoveryToken(TK_Comma);
}

Optional<Token> Parser::missingRightParenthesesOfTupleRecovery() {
  Diagnostic(DiagId::err_missing_right_parenthesis_closing_tuple,
             tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_RParentheses)))
      .print(Out, SM);
  return getRecoveryToken(TK_RParentheses);
}

/// ExprStmt = Expr ";"
/// ExprList = ExprStmt*
Optional<std::vector<source::ExprStmt>> Parser::ExprList() {
  std::vector<source::ExprStmt> result;

  while (tk().isOneOf(/*firstTokensOfExpr*/ TK_IntegralNumber, TK_Ret, TK_Semicolon)) {
    if (tk().is(TK_Semicolon)) {
      // Consume dangling ';'s
      result.push_back(source::ExprStmt(nullptr, consume()));
      continue;
    }

    auto expr = Expr();
    if (!expr) {
      // Errors already reported
      return None;
    }

    auto semicolon =
        parseOrTryRecover<&Parser::ParseToken<TK_Semicolon>,
                          &Parser::missingSemicolonAfterExpressionRecovery>();
    if (!semicolon) {
      return None;
    }

    result.push_back(source::ExprStmt(std::move(expr), *semicolon));
  }

  return result;
}

/// Expr = ExprNumber | "ret" Expr
RAIIExpr Parser::Expr() {
  if (auto num = Number()) {
    return std::make_unique<source::ExprIntegerNumber>(*num);
  }

  if (tk().is(TK_Ret)) {
    Token ret = consume();

    if (tk().is(tmplang::TK_Semicolon)) {
      return std::make_unique<source::ExprRet>(ret);
    }

    auto retExpr = Expr();
    if (!retExpr) {
      Diagnostic(DiagId::err_missing_expression_after_ret_keyword,
                 prevTk().getSpan(), NoHint{})
          .print(Out, SM);
      return nullptr;
    }
    return std::make_unique<source::ExprRet>(ret, std::move(retExpr));
  }

  return nullptr;
}

Optional<Token> Parser::missingSemicolonAfterExpressionRecovery() {
  // 1:
  //   5
  // }  ^___ semicolon here

  // 2:
  //   5
  //   <Start_of_expr>;
  //    ^___ semicolon here
  const bool missingSemicolon = tk().isOneOf(TK_RKeyBracket, TK_IntegralNumber);
  if (!missingSemicolon) {
    return None;
  }

  Diagnostic(
      DiagId::err_missing_semicolon_after_expr, prevTk().getSpan(),
      InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_Semicolon)))
      .print(Out, SM);

  return getRecoveryToken(TK_Semicolon);
}

/// Identifier = [a-zA-Z][a-zA-Z0-9]*;
Optional<Token> Parser::Identifier() {
  return tk().is(TK_Identifier) ? consume() : Optional<Token>{};
}

Optional<Token> Parser::Number() {
  return tk().is(TK_IntegralNumber) ? consume() : Optional<Token>{};
}

const Token &Parser::prevTk() const { return ParserState.PrevToken; }
const Token &Parser::tk() const { return ParserState.CurrToken; }
const Token &Parser::nextTk() const { return ParserState.NextToken; }

SourceLocation Parser::getStartCurrToken() const {
  return tk().getSpan().Start;
}
SourceLocation Parser::getEndCurrToken() const { return tk().getSpan().End; }

bool Parser::emitUnknownTokenDiag(const bool force) const {
  if (force || tk().is(TK_Unknown)) {
    Diagnostic(DiagId::err_found_unknown_token, tk().getSpan(), NoHint())
        .print(Out, SM);
    return true;
  }
  return false;
}

Token Parser::consume() {
  Token token = tk();

  // Rotate tokens
  ParserState.PrevToken = ParserState.CurrToken;
  ParserState.CurrToken = ParserState.NextToken;
  ParserState.NextToken = Lex.next();

  return token;
}

Token Parser::getRecoveryToken(TokenKind kind) {
  ParserState.NumberOfRecoveriesPerformed++;
  return Token(kind, RecoveryLoc, RecoveryLoc, /*isRecovery=*/true);
}

Token Parser::getRecoveryToken(StringRef id, TokenKind kind) {
  ParserState.NumberOfRecoveriesPerformed++;
  return Token(id, kind, RecoveryLoc, RecoveryLoc, /*isRecovery=*/true);
}

Optional<source::CompilationUnit>
tmplang::Parse(tmplang::Lexer &lex, raw_ostream &out, const SourceManager &sm) {
  return Parser(lex, out, sm).Start();
}
