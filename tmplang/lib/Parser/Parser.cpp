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
  std::unique_ptr<source::DataDecl> DataDecl();
  Optional<Token> missingDataDeclId();
  Optional<Token> missingDataDeclStartingEq();

  Optional<SmallVector<source::DataFieldDecl, 4>> DataFieldDeclList();
  Optional<Token> missingCommaDataFieldSepRecovery();

  Optional<source::DataFieldDecl> DataFieldDecl();
  Optional<Token> missingDataFieldId();
  Optional<Token> missingDataFieldColonSep();
  Optional<Token> missingDataFieldType();

  std::unique_ptr<source::SubprogramDecl> SubprogramDecl();
  std::unique_ptr<source::SubprogramDecl> ArrowAndEndOfSubprogramFactored(
      Token funcType, Token id, Optional<Token> colon = None,
      SmallVector<source::ParamDecl, 4> paramList = {});
  Optional<Token> missingSubprogramTypeRecovery();
  Optional<Token> missingSubprogramIdRecovery();
  Optional<Token> missingReturnTypeArrowRecovery();

  Optional<SmallVector<source::ParamDecl, 4>> ParamList();
  Optional<Token> missingCommaParamSepRecovery();

  Optional<source::ParamDecl> Param();
  Optional<Token> missingVariableOnParamRecovery();

  Optional<source::SubprogramDecl::Block> Block();
  Optional<Token> missingLeftKeyBracketRecovery();
  Optional<Token> missingRightKeyBracketRecovery();

  Optional<std::vector<source::ExprStmt>> ExprList();

  RAIIExpr Expr();
  Optional<Token> missingSemicolonAfterExpressionRecovery();

  Optional<Token> Identifier();
  Optional<Token> Number();
  std::unique_ptr<source::ExprAggregateDataAccess>
  ExprAggregateDataAccess(source::ExprAggregateDataAccess::BaseNode baseExpr);
  std::unique_ptr<source::ExprTuple> ExprTuple();
  Optional<Token> missingCommaBetweenTupleElemsRecovery();

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
  std::vector<std::unique_ptr<source::Decl>> decls;
  while (true) {
    if (tk().is(TokenKind::TK_EOF)) {
      return source::CompilationUnit(std::move(decls),
                                     ParserState.NumberOfRecoveriesPerformed);
    }

    if (tk().is(TK_Data)) {
      auto fieldDecl = DataDecl();
      if (!fieldDecl) {
        // Nothing to do, already reported
        return None;
      }
      decls.push_back(std::move(fieldDecl));
      continue;
    }

    if (tk().isOneOf(TK_FnType, TK_ProcType)) {
      auto subprogram = SubprogramDecl();
      if (!subprogram) {
        // Nothing to do, already reported
        return None;
      }
      decls.push_back(std::move(subprogram));
      continue;
    }

    // FIXME: This should be: found X when Y was expected
    Diagnostic(DiagId::err_found_unknown_token, tk().getSpan(), NoHint())
        .print(Out, SM);
    return None;
  }

  return source::CompilationUnit(std::move(decls),
                                 ParserState.NumberOfRecoveriesPerformed);
}

std::unique_ptr<source::SubprogramDecl> Parser::ArrowAndEndOfSubprogramFactored(
    Token funcType, Token id, Optional<Token> colon,
    SmallVector<source::ParamDecl, 4> paramList) {
  if (auto arrow = parseOrTryRecover<&Parser::ParseToken<TK_RArrow>,
                                     &Parser::missingReturnTypeArrowRecovery>(
          /*emitUnexpectedTokenDiag*/ false)) {
    // [1]
    auto returnType = Type();
    if (!returnType) {
      // Nothing to report here, reported on Type
      return nullptr;
    }

    auto block = Block();
    if (!block) {
      // Nothing to report here, reported on Block
      return nullptr;
    }

    return std::make_unique<source::SubprogramDecl>(
        funcType, id, std::move(*block), colon, std::move(paramList),
        source::SubprogramDecl::ArrowAndType{*arrow, std::move(returnType)});
  }

  // [2]
  auto block = Block();
  if (!block) {
    // Nothing to report here, reported on Block
    return nullptr;
  }

  return std::make_unique<source::SubprogramDecl>(
      funcType, id, std::move(*block), colon, std::move(paramList));
}

std::unique_ptr<source::DataDecl> Parser::DataDecl() {
  auto data = ParseToken<TK_Data>();
  if (!data) {
    return nullptr;
  }

  auto id = parseOrTryRecover<&Parser::ParseToken<TK_Identifier>,
                              &Parser::missingDataDeclId>();
  if (!id) {
    return nullptr;
  }

  auto eq = parseOrTryRecover<&Parser::ParseToken<TK_Eq>,
                              &Parser::missingDataDeclStartingEq>();
  if (!eq) {
    return nullptr;
  }

  auto fields = DataFieldDeclList();
  if (!fields) {
    return nullptr;
  }

  auto semicolon = ParseToken<TK_Semicolon>();
  if (!semicolon) {
    // TODO:
    return nullptr;
  }

  return std::make_unique<source::DataDecl>(*data, *id, *eq, std::move(*fields),
                                            *semicolon);
}

Optional<Token> Parser::missingDataDeclId() {
  // data   =
  //      ^___ id here

  const bool missingId = prevTk().is(TK_Data) && tk().is(TK_Eq);
  if (!missingId) {
    return None;
  }

  constexpr StringLiteral placeHolder = "<data_decl_identifier>";
  Diagnostic(DiagId::err_missing_id_of_data_decl, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), placeHolder))
      .print(Out, SM);

  return getRecoveryToken(placeHolder, TK_Identifier);
}

Optional<Token> Parser::missingDataDeclStartingEq() {
  // data Dummy
  //            ^___ '=' id here
  //   a : i32;

  const bool missingId = prevTk().is(TK_Identifier) && tk().is(TK_Identifier);
  if (!missingId) {
    return None;
  }

  Diagnostic(DiagId::err_missing_eq_of_data_decl, prevTk().getSpan(),
             InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_Eq)))
      .print(Out, SM);

  return getRecoveryToken(TK_Eq);
}

Optional<source::DataFieldDecl> Parser::DataFieldDecl() {
  Optional<Token> id = parseOrTryRecover<&Parser::ParseToken<TK_Identifier>,
                                         &Parser::missingDataFieldId>();
  if (!id) {
    return None;
  }

  Optional<Token> colon =
      parseOrTryRecover<&Parser::ParseToken<TK_Colon>,
                        &Parser::missingDataFieldColonSep>();
  if (!colon) {
    return None;
  }

  auto ty = Type();
  if (!ty) {
    // Nothing to report here, reported on Type
    return None;
  }

  return source::DataFieldDecl(*id, *colon, std::move(ty));
}

Optional<Token> Parser::missingDataFieldId() {
  // data Foo =
  //    : i32;
  //  ^___ id here

  const bool missingId = prevTk().is(TK_Eq) && tk().is(TK_Colon);
  if (!missingId) {
    return None;
  }

  constexpr StringLiteral placeHolder = "<data_field_identifier>";

  Diagnostic(DiagId::err_missing_id_of_data_field, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), placeHolder))
      .print(Out, SM);

  return getRecoveryToken(placeHolder, TK_Identifier);
}

Optional<Token> Parser::missingDataFieldColonSep() {
  // data Foo =
  //   a  i32;
  //     ^___ colon here

  const bool missingColon =
      prevTk().is(tmplang::TK_Identifier) &&
      tk().isOneOf(tmplang::TK_Identifier, tmplang::TK_LParentheses);
  if (!missingColon) {
    return None;
  }

  Diagnostic(DiagId::err_missing_colon_on_data_field, prevTk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Colon)))
      .print(Out, SM);

  return getRecoveryToken(TK_Colon);
}

Optional<Token> Parser::missingDataFieldType() {
  // data Foo =
  //   a  i32;
  //     ^___ colon here

  const bool missingDataType =
      prevTk().is(tmplang::TK_Colon) &&
      tk().isOneOf(tmplang::TK_Comma, tmplang::TK_Semicolon);
  if (!missingDataType) {
    return None;
  }

  Diagnostic(DiagId::err_missing_type_on_data_field, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  // FIXME: Is this correct? A types is expected, TK_Identifier can work
  // to represent an identifier
  return getRecoveryToken(TK_Identifier);
}

/// DataFieldDeclList = DataFieldDecl (",", DataFieldDecl)*;
Optional<SmallVector<source::DataFieldDecl, 4>> Parser::DataFieldDeclList() {
  SmallVector<source::DataFieldDecl, 4> dataFieldDecls;

  auto firstParam = DataFieldDecl();
  if (!firstParam) {
    // Nothing to report here, reported on Param
    return None;
  }
  dataFieldDecls.push_back(std::move(*firstParam));

  while (auto comma =
             parseOrTryRecover<&Parser::ParseToken<TK_Comma>,
                               &Parser::missingCommaDataFieldSepRecovery>(
                 /*emitUnexpectedTokenDiag*/ false)) {
    dataFieldDecls.back().setComma(*comma);

    auto dataField = DataFieldDecl();
    if (!dataField) {
      // Nothing to report here, reported on Param
      return None;
    }

    dataFieldDecls.push_back(std::move(*dataField));
  }

  return dataFieldDecls;
}

Optional<Token> Parser::missingCommaDataFieldSepRecovery() {
  // data Foo =
  //   a : i32
  //          ^___ comma here
  //   b : i32,;

  const bool missingComma = tk().isOneOf(TK_Identifier) &&
                            // TokenKinds corresponding to end of Type
                            prevTk().isOneOf(TK_Identifier, TK_RParentheses);
  if (!missingComma) {
    return None;
  }

  Diagnostic(DiagId::err_missing_comma_between_data_fields, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  return getRecoveryToken(TK_Comma);
}

/// Function_type = "proc" | "fn";

/// Function_Definition =
///  [1] | Function_Type, Identifier, ":", Param_List, "->", Type, Block
///  [2] | Function_Type, Identifier, ":", Param_List, Block
///  [3] | Function_Type, Identifier, "->", Type, Block
///  [4] | Function_Type, Identifier, Block;
std::unique_ptr<source::SubprogramDecl> Parser::SubprogramDecl() {
  auto funcType = Parser::ParseToken<TK_FnType, TK_ProcType>();
  if (!funcType) {
    // Since this is the start of top level declaration, lets consume the
    // unknowns until we find something we understand
    if (tk().is(tmplang::TK_Unknown)) {
      consume();
      return SubprogramDecl();
    }
    return nullptr;
  }

  auto id = parseOrTryRecover<&Parser::Identifier,
                              &Parser::missingSubprogramIdRecovery>();
  if (!id) {
    return nullptr;
  }

  // [1] && [2]
  if (tk().is(TK_Colon)) {
    auto colon = consume();

    auto paramList = ParamList();
    if (!paramList) {
      // Nothing to report here, reported on ParamList
      return nullptr;
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
Optional<source::SubprogramDecl::Block> Parser::Block() {
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

  return source::SubprogramDecl::Block{*lKeyBracket, std::move(*exprs),
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

  if (tk().isOneOf(/*firstTokensOfTypeExceptRet*/ TK_LParentheses,
                   TK_IntegerNumber, TK_Identifier)) {
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

  while (tk().isOneOf(/*firstTokensOfExpr*/ TK_LParentheses, TK_IntegerNumber,
                      TK_Ret, TK_Identifier)) {
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

std::unique_ptr<source::ExprAggregateDataAccess>
Parser::ExprAggregateDataAccess(
    source::ExprAggregateDataAccess::BaseNode baseExpr) {
  while (tk().is(TK_Dot)) {
    auto dot = consume();
    Optional<tmplang::Token> idOrNum = Identifier();
    if (!idOrNum) {
      idOrNum = Number();
      if (!idOrNum) {
        // FIXME: Add token recovery and diags error
        return nullptr;
      }
    }
    baseExpr = std::make_unique<source::ExprAggregateDataAccess>(
        std::move(baseExpr), dot, *idOrNum);
  }
  assert(
      std::holds_alternative<std::unique_ptr<source::ExprAggregateDataAccess>>(
          baseExpr));

  return std::move(
      std::get<std::unique_ptr<source::ExprAggregateDataAccess>>(baseExpr));
}

/// Expr = ExprNumber | "ret" Expr | ExprTuple | ExprVarRef | ExprAggregateDataAccess
RAIIExpr Parser::Expr() {
  if (auto id = Identifier()) {
    if (tk().isNot(TK_Dot)) {
      return std::make_unique<source::ExprVarRef>(*id);
    }
    return ExprAggregateDataAccess(source::ExprVarRef(*id));
  }

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

  if (tk().is(TK_LParentheses)) {
    auto tupleExpr = ExprTuple();
    if (!tupleExpr) {
      return nullptr;
    }

    if (tk().isNot(TK_Dot)) {
      return std::move(tupleExpr);
    }
    return ExprAggregateDataAccess(std::move(tupleExpr));
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
  const bool missingSemicolon = tk().isOneOf(TK_RKeyBracket, TK_IntegerNumber);
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
  return tk().is(TK_IntegerNumber) ? consume() : Optional<Token>{};
}

std::unique_ptr<source::ExprTuple> Parser::ExprTuple() {
  Token lparentheses = consume();
  assert(lparentheses.is(TK_LParentheses) &&
         "This is validated on the call-site");

  SmallVector<source::TupleElem, 4> vals;

  if (tk().isOneOf(/*firstTokensOfExpr*/ TK_LParentheses, TK_IntegerNumber,
                   TK_Identifier)) {
    auto firstExpr = Expr();
    if (!firstExpr) {
      // Nothing to report here, reported on Expr
      return nullptr;
    }

    vals.push_back(std::move(firstExpr));

    while (
        auto comma =
            parseOrTryRecover<&Parser::ParseToken<TK_Comma>,
                              &Parser::missingCommaBetweenTupleElemsRecovery>(
                /*emitUnexpectedTokenDiag*/ false)) {
      auto followingVal = Expr();
      if (!followingVal) {
        // Nothing to do here, reported on Expr
        return nullptr;
      }

      vals.back().setComma(std::move(*comma));
      vals.push_back(std::move(followingVal));
    }
  }

  auto rparentheses =
      parseOrTryRecover<&Parser::ParseToken<TK_RParentheses>,
                        &Parser::missingRightParenthesesOfTupleRecovery>();
  assert(rparentheses && "This is unconditionally valid");

  return std::make_unique<source::ExprTuple>(lparentheses, std::move(vals),
                                             *rparentheses);
}

Optional<Token> Parser::missingCommaBetweenTupleElemsRecovery() {
  // fn foo: (() 3);
  // fn foo: (5  3);
  // fn foo: (5  ());
  // fn foo: (5  3);
  //            ^___ comma here

  // But not in the following cases:
  //   fn foo: (i32  var {}
  const bool missingComma =
      prevTk().isOneOf(TK_IntegerNumber, TK_RParentheses) &&
      tk().isOneOf(TK_IntegerNumber, TK_LParentheses) &&
      (nextTk().isNot(TK_IntegerNumber) && nextTk().isNot(TK_LKeyBracket));
  if (!missingComma) {
    return None;
  }

  Diagnostic(DiagId::err_missing_comma_prior_tuple_param, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  return getRecoveryToken(TK_Comma);
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
