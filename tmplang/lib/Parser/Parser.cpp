#include <tmplang/Parser/Parser.h>

#include <tmplang/Diagnostics/Diagnostic.h>
#include <tmplang/Diagnostics/Hint.h>
#include <tmplang/Tree/Source/Decls.h>
#include <tmplang/Tree/Source/Exprs.h>
#include <tmplang/Tree/Source/Types.h>
#include <utility>

using namespace tmplang;

namespace {

class Parser {
public:
  Parser(Lexer &lex, raw_ostream &out, const SourceManager &sm)
      : Lex(lex), Out(out), SM(sm) {
    // These two "consume" calls will initialize the current and next token
    consume();
    consume();
  }

  std::optional<source::CompilationUnit> Start();

private:
  std::unique_ptr<source::DataDecl> DataDecl();
  std::optional<Token> missingDataDeclId();
  std::optional<Token> missingDataDeclStartingEq();

  std::unique_ptr<source::UnionDecl> UnionDecl();

  std::optional<source::UnionAlternativeDecl> UnionAlternativeDecl();
  /// DataFieldDeclList = DataFieldDecl (",", DataFieldDecl)*;
  std::optional<source::OneElementOrMoreList<source::UnionAlternativeDecl>>
  UnionAlternativeDeclList() {
    return ParseOneOrMoreCommaSeparatedItemList<
        source::UnionAlternativeDecl, &Parser::UnionAlternativeDecl>();
  }

  std::optional<source::UnionAlternativeFieldDecl> UnionAlternativeFieldDecl();
  /// DataFieldDeclList = DataFieldDecl (",", DataFieldDecl)*;
  std::optional<source::OneElementOrMoreList<source::UnionAlternativeFieldDecl>>
  UnionAlternativeFieldDeclList() {
    return ParseOneOrMoreCommaSeparatedItemList<
        source::UnionAlternativeFieldDecl,
        &Parser::UnionAlternativeFieldDecl>();
  }

  /// DataFieldDeclList = DataFieldDecl (",", DataFieldDecl)*;
  std::optional<source::OneElementOrMoreList<source::DataFieldDecl>>
  DataFieldDeclList() {
    return ParseOneOrMoreCommaSeparatedItemList<source::DataFieldDecl,
                                                &Parser::DataFieldDecl>();
  }

  std::optional<source::DataFieldDecl> DataFieldDecl();
  std::optional<Token> missingDataFieldId();
  std::optional<Token> missingDataFieldColonSep();
  std::optional<Token> missingDataFieldType();

  std::unique_ptr<source::SubprogramDecl> SubprogramDecl();
  std::unique_ptr<source::SubprogramDecl> ArrowAndEndOfSubprogramFactored(
      Token funcType, Token id,
      std::optional<SpecificToken<TK_Colon>> colon = nullopt,
      source::MinElementList<source::ParamDecl, 0> paramList = {});
  std::optional<Token> missingSubprogramTypeRecovery();
  std::optional<Token> missingSubprogramIdRecovery();
  std::optional<Token> missingReturnTypeArrowRecovery();

  /// Param_List = (Param (",", Param)*)?;
  std::optional<source::VariadicList<source::ParamDecl>> ParamList() {
    return ParseVariadicCommaSeparatedItemList<source::ParamDecl,
                                               &Parser::Param>();
  }

  std::optional<source::ParamDecl> Param();
  std::optional<Token> missingVariableOnParamRecovery();

  std::optional<source::SubprogramDecl::Block> Block();
  std::optional<Token> missingLeftKeyBracketRecovery();
  std::optional<Token> missingRightKeyBracketRecovery();

  std::optional<std::vector<source::ExprStmt>> ExprList();

  std::unique_ptr<source::Expr> ConstantExpr();
  std::unique_ptr<source::Expr> Expr();
  std::optional<Token> missingSemicolonAfterExpressionRecovery();

  std::optional<Token> Identifier();
  std::optional<Token> Number();
  std::unique_ptr<source::ExprAggregateDataAccess>
  ExprAggregateDataAccess(source::ExprAggregateDataAccess::BaseNode baseExpr);

  std::optional<source::TupleDestructuration> TupleDestructuration();
  std::optional<source::DataDestructurationElem>
  DataDestructurationFieldNameAndValue();
  std::optional<source::DataDestructuration> DataDestructuration();
  std::unique_ptr<source::ExprMatchCaseLhsVal> MatchCaseLhsValue();
  std::optional<source::ExprMatchCase::Lhs> MatchCaseLhs();
  std::optional<source::ExprMatchCase> MatchCase();
  std::unique_ptr<source::ExprMatch> ExprMatch();

  std::unique_ptr<source::ExprTuple> ExprTuple();
  std::optional<Token> missingCommaBetweenTupleElemsRecovery();

  // Types...
  source::RAIIType Type();
  source::RAIIType NamedType();
  source::RAIIType TupleType();
  std::optional<Token> missingCommaBetweenTypesRecovery();
  std::optional<Token> missingRightParenthesesOfTupleRecovery();

  // Simple token parsing. This function is useful in conjuntion with
  // parseOrRecover so it can recieve the address
  template <TokenKind... Kinds> std::optional<Token> ParseToken() {
    if (tk().isOneOf(Kinds...)) {
      return consume();
    }
    return nullopt;
  }

  template <typename T, unsigned MinSize,
            std::optional<T> (Parser::*itemParserFunc)()>
  std::optional<source::MinElementList<T, MinSize>>
  ParseItemCommaSeparatedList() {
    typename source::MinElementList<T, MinSize>::InternalList_t items;

    auto firstItem = (this->*itemParserFunc)();
    if (!firstItem) {
      return std::nullopt;
    }
    items.push_back(std::move(*firstItem));

    while (auto comma = Parser::ParseToken<TK_Comma>()) {
      items.back().setComma(std::move(*comma));

      auto restItems = (this->*itemParserFunc)();
      if (!restItems) {
        return std::nullopt;
      }

      items.push_back(std::move(*restItems));
    }

    return std::optional<source::MinElementList<T, MinSize>>(std::in_place,
                                                             std::move(items));
  }

  /// Parses a guaranteed one sized or more list of elements using the provided
  /// function \ref itemParserFunc. This functions relies on nodes inheriting
  /// the TrailingComma trait.
  template <typename T, std::optional<T> (Parser::*itemParserFunc)()>
  std::optional<source::OneElementOrMoreList<T>>
  ParseOneOrMoreCommaSeparatedItemList() {
    return ParseItemCommaSeparatedList<T, 1, itemParserFunc>();
  }

  /// Parses a variadic size list of elements using the provided function \ref
  /// itemParserFunc. This functions relies on nodes inheriting the
  /// TrailingComma trait.
  template <typename T, std::optional<T> (Parser::*itemParserFunc)()>
  std::optional<source::VariadicList<T>> ParseVariadicCommaSeparatedItemList() {
    return ParseItemCommaSeparatedList<T, 0, itemParserFunc>();
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

  template <std::optional<Token> (Parser::*parsingFunc)(),
            std::optional<Token> (Parser::*recoveryFunc)() = nullptr>
  std::optional<Token> parseOrTryRecover(bool emitUnexpectedTokenDiag = true) {
    // Try the normal path, parse as it is expected
    if (auto parsedToken = (this->*parsingFunc)()) {
      return *parsedToken;
    }

    // If there is no recovery function, do not recover
    if (recoveryFunc == nullptr) {
      return nullopt;
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

    return nullopt;
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
std::optional<source::CompilationUnit> Parser::Start() {
  std::vector<std::unique_ptr<source::Decl>> decls;
  while (true) {
    if (tk().is(TokenKind::TK_EOF)) {
      return source::CompilationUnit(std::move(decls),
                                     ParserState.NumberOfRecoveriesPerformed);
    }

    if (tk().is(TK_Union)) {
      auto unionDecl = UnionDecl();
      if (!unionDecl) {
        // Nothing to do, already reported
        return nullopt;
      }
      decls.push_back(std::move(unionDecl));
      continue;
    }

    if (tk().is(TK_Data)) {
      auto fieldDecl = DataDecl();
      if (!fieldDecl) {
        // Nothing to do, already reported
        return nullopt;
      }
      decls.push_back(std::move(fieldDecl));
      continue;
    }

    if (tk().isOneOf(TK_FnType, TK_ProcType)) {
      auto subprogram = SubprogramDecl();
      if (!subprogram) {
        // Nothing to do, already reported
        return nullopt;
      }
      decls.push_back(std::move(subprogram));
      continue;
    }

    // FIXME: This should be: found X when Y was expected
    Diagnostic(DiagId::err_found_unknown_token, tk().getSpan(), NoHint())
        .print(Out, SM);
    return nullopt;
  }

  return source::CompilationUnit(std::move(decls),
                                 ParserState.NumberOfRecoveriesPerformed);
}

std::unique_ptr<source::SubprogramDecl> Parser::ArrowAndEndOfSubprogramFactored(
    Token funcType, Token id, std::optional<SpecificToken<TK_Colon>> colon,
    source::VariadicList<source::ParamDecl> paramList) {
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
        std::move(funcType), std::move(id), std::move(*block), std::move(colon),
        std::move(paramList),
        source::SubprogramDecl::ArrowAndType{std::move(*arrow),
                                             std::move(returnType)});
  }

  // [2]
  auto block = Block();
  if (!block) {
    // Nothing to report here, reported on Block
    return nullptr;
  }

  return std::make_unique<source::SubprogramDecl>(
      std::move(funcType), std::move(id), std::move(*block), std::move(colon),
      std::move(paramList));
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

  return std::make_unique<source::DataDecl>(std::move(*data), std::move(*id),
                                            std::move(*eq), std::move(*fields),
                                            std::move(*semicolon));
}

std::optional<Token> Parser::missingDataDeclId() {
  // data   =
  //      ^___ id here

  const bool missingId = prevTk().is(TK_Data) && tk().is(TK_Eq);
  if (!missingId) {
    return nullopt;
  }

  constexpr StringLiteral placeHolder = "<data_decl_identifier>";
  Diagnostic(DiagId::err_missing_id_of_data_decl, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), placeHolder))
      .print(Out, SM);

  return getRecoveryToken(placeHolder, TK_Identifier);
}

std::optional<Token> Parser::missingDataDeclStartingEq() {
  // data Dummy
  //            ^___ '=' id here
  //   a : i32;

  const bool missingId = prevTk().is(TK_Identifier) && tk().is(TK_Identifier);
  if (!missingId) {
    return nullopt;
  }

  Diagnostic(DiagId::err_missing_eq_of_data_decl, prevTk().getSpan(),
             InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_Eq)))
      .print(Out, SM);

  return getRecoveryToken(TK_Eq);
}

std::unique_ptr<source::UnionDecl> Parser::UnionDecl() {
  auto unionKeyword = Parser::ParseToken<TK_Union>();
  if (!unionKeyword) {
    // TODO: Emmit error and recover
    return nullptr;
  }

  auto id = Parser::ParseToken<TK_Identifier>();
  if (!id) {
    // TODO: Emmit error and recover
    return nullptr;
  }

  auto eq = Parser::ParseToken<TK_Eq>();
  if (!eq) {
    // TODO: Emmit error and recover
    return nullptr;
  }

  auto alternatives = UnionAlternativeDeclList();
  if (!alternatives) {
    return nullptr;
  }

  auto semicolon = Parser::ParseToken<TK_Semicolon>();
  if (!semicolon) {
    // TODO: Emmit error and recover
    return nullptr;
  }

  return std::make_unique<source::UnionDecl>(
      std::move(*unionKeyword), std::move(*id), std::move(*eq),
      std::move(*alternatives), std::move(*semicolon));
}

std::optional<source::UnionAlternativeDecl> Parser::UnionAlternativeDecl() {
  auto id = Parser::ParseToken<TK_Identifier>();
  if (!id) {
    // TODO: Emmit error and recover
    return std::nullopt;
  }

  auto lParen = Parser::ParseToken<TK_LParentheses>();
  if (!lParen) {
    // TODO: Emmit error and recover
    return std::nullopt;
  }

  auto fields = UnionAlternativeFieldDeclList();
  if (!fields) {
    return std::nullopt;
  }

  auto rParen = Parser::ParseToken<TK_RParentheses>();
  if (!rParen) {
    // TODO: Emmit error and recover
    return std::nullopt;
  }

  return source::UnionAlternativeDecl(std::move(*id), std::move(*lParen),
                                      std::move(*fields), std::move(*rParen));
}

std::optional<source::UnionAlternativeFieldDecl>
Parser::UnionAlternativeFieldDecl() {
  auto id = Parser::ParseToken<TK_Identifier>();
  if (!id) {
    // TODO: Emmit error and recover
    return std::nullopt;
  }

  auto colon = Parser::ParseToken<TK_Colon>();
  if (!colon) {
    // TODO: Emmit error and recover
    return std::nullopt;
  }

  auto type = Type();
  if (!type) {
    return std::nullopt;
  }

  return source::UnionAlternativeFieldDecl(std::move(*id), std::move(*colon),
                                           std::move(type));
}

std::optional<source::DataFieldDecl> Parser::DataFieldDecl() {
  std::optional<Token> id =
      parseOrTryRecover<&Parser::ParseToken<TK_Identifier>,
                        &Parser::missingDataFieldId>();
  if (!id) {
    return nullopt;
  }

  std::optional<Token> colon =
      parseOrTryRecover<&Parser::ParseToken<TK_Colon>,
                        &Parser::missingDataFieldColonSep>();
  if (!colon) {
    return nullopt;
  }

  auto ty = Type();
  if (!ty) {
    // Nothing to report here, reported on Type
    return nullopt;
  }

  return source::DataFieldDecl(std::move(*id), *std::move(colon),
                               std::move(ty));
}

std::optional<Token> Parser::missingDataFieldId() {
  // data Foo =
  //    : i32;
  //  ^___ id here

  const bool missingId = prevTk().is(TK_Eq) && tk().is(TK_Colon);
  if (!missingId) {
    return nullopt;
  }

  constexpr StringLiteral placeHolder = "<data_field_identifier>";

  Diagnostic(DiagId::err_missing_id_of_data_field, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), placeHolder))
      .print(Out, SM);

  return getRecoveryToken(placeHolder, TK_Identifier);
}

std::optional<Token> Parser::missingDataFieldColonSep() {
  // data Foo =
  //   a  i32;
  //     ^___ colon here

  const bool missingColon =
      prevTk().is(tmplang::TK_Identifier) &&
      tk().isOneOf(tmplang::TK_Identifier, tmplang::TK_LParentheses);
  if (!missingColon) {
    return nullopt;
  }

  Diagnostic(DiagId::err_missing_colon_on_data_field, prevTk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Colon)))
      .print(Out, SM);

  return getRecoveryToken(TK_Colon);
}

std::optional<Token> Parser::missingDataFieldType() {
  // data Foo =
  //   a  i32;
  //     ^___ colon here

  const bool missingDataType =
      prevTk().is(tmplang::TK_Colon) &&
      tk().isOneOf(tmplang::TK_Comma, tmplang::TK_Semicolon);
  if (!missingDataType) {
    return nullopt;
  }

  Diagnostic(DiagId::err_missing_type_on_data_field, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  // FIXME: Is this correct? A types is expected, TK_Identifier can work
  // to represent an identifier
  return getRecoveryToken(TK_Identifier);
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

    return ArrowAndEndOfSubprogramFactored(
        *funcType, *id, SpecificToken<TK_Colon>(std::move(colon)),
        std::move(*paramList));
  }

  return ArrowAndEndOfSubprogramFactored(*funcType, *id);
}

std::optional<Token> Parser::missingSubprogramTypeRecovery() {
  const bool potentialStartOfSubprogram =
      tk().is(TK_Identifier) &&
      nextTk().isOneOf(TK_Colon, TK_RArrow, TK_LKeyBracket);
  if (!potentialStartOfSubprogram) {
    return nullopt;
  }

  Diagnostic(DiagId::err_missing_subprogram_class, tk().getSpan(),
             InsertTextAtHint(tk().getSpan().Start,
                              {ToString(TK_FnType), ToString(TK_ProcType)}, ""))
      .print(Out, SM);

  return getRecoveryToken(TK_FnType);
}

std::optional<Token> Parser::missingSubprogramIdRecovery() {
  const bool missingId = prevTk().isOneOf(TK_FnType, TK_ProcType) &&
                         tk().isOneOf(TK_Colon, TK_RArrow, TK_LKeyBracket);
  if (!missingId) {
    return nullopt;
  }

  constexpr StringLiteral placeHolder = "<subprogram_identifier>";

  Diagnostic(DiagId::err_missing_subprogram_id, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), placeHolder))
      .print(Out, SM);

  return getRecoveryToken(placeHolder, TK_Identifier);
}

std::optional<Token> Parser::missingReturnTypeArrowRecovery() {
  const bool missingArrow =
      tk().isOneOf(/*firstTokensOfType*/ TK_Identifier, TK_LParentheses);
  if (!missingArrow) {
    return nullopt;
  }

  Diagnostic(DiagId::err_missing_arrow, prevTk().getSpan(),
             InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_RArrow)))
      .print(Out, SM);

  return getRecoveryToken(TK_RArrow);
}

/// Param = Type Identifier;
std::optional<source::ParamDecl> Parser::Param() {
  auto type = Type();
  if (!type) {
    // Nothing to report here, reported on Type
    return nullopt;
  }

  auto paramId = parseOrTryRecover<&Parser::Identifier,
                                   &Parser::missingVariableOnParamRecovery>();
  if (!paramId) {
    return nullopt;
  }

  return source::ParamDecl(std::move(type), std::move(*paramId));
}

std::optional<Token> Parser::missingVariableOnParamRecovery() {
  constexpr StringLiteral paramId = "<parameter_id>";

  Diagnostic(DiagId::err_missing_variable_identifier_after_type, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), paramId))
      .print(Out, SM);

  return getRecoveryToken(paramId, TK_Identifier);
}

/// Block = "{" ExprList "}";
std::optional<source::SubprogramDecl::Block> Parser::Block() {
  auto lKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_LKeyBracket>,
                        &Parser::missingLeftKeyBracketRecovery>();
  if (!lKeyBracket) {
    return nullopt;
  }

  std::optional<std::vector<source::ExprStmt>> exprs = ExprList();
  if (!exprs) {
    // Errors already reported
    return nullopt;
  }

  auto rKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_RKeyBracket>,
                        &Parser::missingRightKeyBracketRecovery>();
  if (!rKeyBracket) {
    return nullopt;
  }

  return source::SubprogramDecl::Block{
      std::move(*lKeyBracket), std::move(*exprs), std::move(*rKeyBracket)};
}

std::optional<Token> Parser::missingLeftKeyBracketRecovery() {
  Diagnostic(
      DiagId::err_missing_left_key_brace, prevTk().getSpan(),
      InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_LKeyBracket)))
      .print(Out, SM);

  return getRecoveryToken(TK_LKeyBracket);
}

std::optional<Token> Parser::missingRightKeyBracketRecovery() {
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

  return source::make_RAIIType<source::NamedType>(std::move(*id));
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
      std::move(lparentheses), std::move(types), std::move(commas),
      std::move(*rparentheses));
}

std::optional<Token> Parser::missingCommaBetweenTypesRecovery() {
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
    return nullopt;
  }

  Diagnostic(DiagId::err_missing_comma_prior_tuple_param, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  return getRecoveryToken(TK_Comma);
}

std::optional<Token> Parser::missingRightParenthesesOfTupleRecovery() {
  Diagnostic(DiagId::err_missing_right_parenthesis_closing_tuple,
             tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_RParentheses)))
      .print(Out, SM);
  return getRecoveryToken(TK_RParentheses);
}

/// ExprStmt = Expr ";"
/// ExprList = ExprStmt*
std::optional<std::vector<source::ExprStmt>> Parser::ExprList() {
  std::vector<source::ExprStmt> result;

  while (tk().isOneOf(/*firstTokensOfExpr*/ TK_LParentheses, TK_IntegerNumber,
                      TK_Ret, TK_Identifier, TK_Match)) {
    if (tk().is(TK_Semicolon)) {
      // Consume dangling ';'s
      result.push_back(source::ExprStmt(nullptr, consume()));
      continue;
    }

    auto expr = Expr();
    if (!expr) {
      // Errors already reported
      return nullopt;
    }

    auto semicolon =
        parseOrTryRecover<&Parser::ParseToken<TK_Semicolon>,
                          &Parser::missingSemicolonAfterExpressionRecovery>();
    if (!semicolon) {
      return nullopt;
    }

    result.push_back(source::ExprStmt(std::move(expr), std::move(*semicolon)));
  }

  return result;
}

std::unique_ptr<source::ExprAggregateDataAccess>
Parser::ExprAggregateDataAccess(
    source::ExprAggregateDataAccess::BaseNode baseExpr) {
  while (tk().is(TK_Dot)) {
    auto dot = consume();
    std::optional<tmplang::Token> idOrNum = Identifier();
    if (!idOrNum) {
      idOrNum = Number();
      if (!idOrNum) {
        // FIXME: Add token recovery and diags error
        return nullptr;
      }
    }
    baseExpr = std::make_unique<source::ExprAggregateDataAccess>(
        std::move(baseExpr), std::move(dot), std::move(*idOrNum));
  }
  assert(
      std::holds_alternative<std::unique_ptr<source::ExprAggregateDataAccess>>(
          baseExpr));

  return std::move(
      std::get<std::unique_ptr<source::ExprAggregateDataAccess>>(baseExpr));
}

/// ConstantExpr = ExprNumber | ToBeAdded(StringLiteral, ...)
std::unique_ptr<source::Expr> Parser::ConstantExpr() {
  if (auto num = Number()) {
    return std::make_unique<source::ExprIntegerNumber>(std::move(*num));
  }
  return nullptr;
}

// clang-format off
/// Expr = ExprNumber | "ret" Expr | ExprTuple | ExprVarRef | ExprAggregateDataAccess
// clang-format on
std::unique_ptr<source::Expr> Parser::Expr() {
  if (auto id = Identifier()) {
    if (tk().isNot(TK_Dot)) {
      return std::make_unique<source::ExprVarRef>(std::move(*id));
    }
    return ExprAggregateDataAccess(source::ExprVarRef(std::move(*id)));
  }

  if (auto num = Number()) {
    return std::make_unique<source::ExprIntegerNumber>(std::move(*num));
  }

  if (tk().is(TK_Match)) {
    return ExprMatch();
  }

  if (tk().is(TK_Ret)) {
    Token ret = consume();

    if (tk().is(tmplang::TK_Semicolon)) {
      return std::make_unique<source::ExprRet>(std::move(ret));
    }

    auto retExpr = Expr();
    if (!retExpr) {
      Diagnostic(DiagId::err_missing_expression_after_ret_keyword,
                 prevTk().getSpan(), NoHint{})
          .print(Out, SM);
      return nullptr;
    }
    return std::make_unique<source::ExprRet>(std::move(ret),
                                             std::move(retExpr));
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

std::optional<Token> Parser::missingSemicolonAfterExpressionRecovery() {
  // 1:
  //   5
  // }  ^___ semicolon here

  // 2:
  //   5
  //   <Start_of_expr>;
  //    ^___ semicolon here
  const bool missingSemicolon = tk().isOneOf(TK_RKeyBracket, TK_IntegerNumber);
  if (!missingSemicolon) {
    return nullopt;
  }

  Diagnostic(
      DiagId::err_missing_semicolon_after_expr, prevTk().getSpan(),
      InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_Semicolon)))
      .print(Out, SM);

  return getRecoveryToken(TK_Semicolon);
}

/// Identifier = [a-zA-Z][a-zA-Z0-9]*;
std::optional<Token> Parser::Identifier() {
  return tk().is(TK_Identifier) ? consume() : std::optional<Token>{};
}

std::optional<Token> Parser::Number() {
  return tk().is(TK_IntegerNumber) ? consume() : std::optional<Token>{};
}

std::optional<source::TupleDestructuration> Parser::TupleDestructuration() {
  auto lparen = parseOrTryRecover<&Parser::ParseToken<TK_LParentheses>>();
  if (!lparen) {
    // TODO: Report error and try recover
    return std::nullopt;
  }

  std::unique_ptr<source::ExprMatchCaseLhsVal> lhsVal = MatchCaseLhsValue();
  if (!lhsVal) {
    return std::nullopt;
  }

  std::vector<source::TupleDestructurationElem> tupleElems;
  while (tk().isNot(TK_RParentheses)) {
    std::optional<Token> comma =
        parseOrTryRecover<&Parser::ParseToken<TK_Comma>>();
    if (!comma) {
      // TODO: Report error and try recover
      return std::nullopt;
    }

    tupleElems.emplace_back(std::move(lhsVal));
    tupleElems.back().setComma(std::move(*comma));

    lhsVal = MatchCaseLhsValue();
    if (!lhsVal) {
      // Nothing to do here, reported on Expr
      return std::nullopt;
    }
  }
  tupleElems.emplace_back(std::move(lhsVal));

  // Already checked to break the above while
  auto rparen = consume();

  return source::TupleDestructuration{std::move(*lparen), std::move(tupleElems),
                                      std::move(rparen)};
}

std::optional<source::DataDestructurationElem>
Parser::DataDestructurationFieldNameAndValue() {
  auto id = parseOrTryRecover<&Parser::ParseToken<TK_Identifier>>();
  if (!id) {
    // TODO: Report error and try recover
    return std::nullopt;
  }
  auto colon = parseOrTryRecover<&Parser::ParseToken<TK_Colon>>();
  if (!colon) {
    // TODO: Report error and try recover
    return std::nullopt;
  }
  auto matchCaseLhsVal = MatchCaseLhsValue();
  if (!matchCaseLhsVal) {
    // Nothing to do here, reported on MatchCaseLhsValue
    return std::nullopt;
  }
  return source::DataDestructurationElem{std::move(*id), std::move(*colon),
                                         std::move(matchCaseLhsVal)};
}

std::optional<source::DataDestructuration> Parser::DataDestructuration() {
  auto lbracket = parseOrTryRecover<&Parser::ParseToken<TK_LKeyBracket>>();
  if (!lbracket) {
    // TODO: Report error and try recover
    return std::nullopt;
  }

  auto fieldNameAndVal = DataDestructurationFieldNameAndValue();
  if (!fieldNameAndVal) {
    return std::nullopt;
  }

  std::vector<source::DataDestructurationElem> dataElems;
  while (tk().isNot(TK_RKeyBracket)) {
    std::optional<Token> comma =
        parseOrTryRecover<&Parser::ParseToken<TK_Comma>>();
    if (!comma) {
      // TODO: Report error and try recover
      return std::nullopt;
    }

    dataElems.emplace_back(std::move(*fieldNameAndVal));
    dataElems.back().setComma(std::move(*comma));

    fieldNameAndVal = DataDestructurationFieldNameAndValue();
    if (!fieldNameAndVal) {
      // Nothing to do here, reported on Expr
      return std::nullopt;
    }
  }
  dataElems.emplace_back(std::move(*fieldNameAndVal));

  // Already checked to break the above while
  auto rparen = consume();

  return source::DataDestructuration{std::move(*lbracket), std::move(dataElems),
                                     std::move(rparen)};
}

std::unique_ptr<source::ExprMatchCaseLhsVal> Parser::MatchCaseLhsValue() {
  if (tk().is(TK_Underscore)) {
    return std::make_unique<source::ExprMatchCaseLhsVal>(
        SpecificToken<TK_Underscore>(consume()));
  }

  if (tk().is(TK_Identifier)) {
    return std::make_unique<source::ExprMatchCaseLhsVal>(
        source::PlaceholderDecl(consume()));
  }

  if (tk().is(TK_LKeyBracket)) {
    auto destructuration = DataDestructuration();
    if (!destructuration) {
      return nullptr;
    }
    return std::make_unique<source::ExprMatchCaseLhsVal>(
        std::move(*destructuration));
  }

  if (tk().is(TK_LParentheses)) {
    auto destructuration = TupleDestructuration();
    if (!destructuration) {
      return nullptr;
    }
    return std::make_unique<source::ExprMatchCaseLhsVal>(
        std::move(*destructuration));
  }

  // If not, try with any constant expression
  if (auto constantExpr = ConstantExpr()) {
    return std::make_unique<source::ExprMatchCaseLhsVal>(
        std::move(constantExpr));
  }

  // TODO: Report error about invalid token
  return nullptr;
}

std::optional<source::ExprMatchCase::Lhs> Parser::MatchCaseLhs() {
  if (tk().is(TK_Otherwise)) {
    return source::ExprMatchCase::Lhs(SpecificToken<TK_Otherwise>(consume()));
  }

  auto lhsVal = MatchCaseLhsValue();
  if (!lhsVal) {
    return std::nullopt;
  }
  return source::ExprMatchCase::Lhs(std::move(lhsVal));
}

std::optional<source::ExprMatchCase> Parser::MatchCase() {
  std::optional<source::ExprMatchCase::Lhs> caseLhsVal = MatchCaseLhs();
  if (!caseLhsVal) {
    return std::nullopt;
  }

  std::optional<Token> arrow =
      parseOrTryRecover<&Parser::ParseToken<TK_RArrow>>();
  if (!arrow) {
    // TODO: Report error and recover
    return std::nullopt;
  }

  auto rhsExpr = Expr();
  if (!rhsExpr) {
    return std::nullopt;
  }

  return source::ExprMatchCase(std::move(*caseLhsVal), std::move(*arrow),
                               std::move(rhsExpr));
}

std::unique_ptr<source::ExprMatch> Parser::ExprMatch() {
  SpecificToken<TK_Match> match = consume();

  auto expr = Expr();
  if (!expr) {
    return nullptr;
  }

  std::optional<Token> lKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_LKeyBracket>>();
  if (!lKeyBracket) {
    // TODO: Report error and recover
    return nullptr;
  }

  SmallVector<source::ExprMatchCase, 4> cases;
  auto matchCase = MatchCase();
  if (!matchCase) {
    // Already reported
    return nullptr;
  }

  while (tk().isNot(TK_RKeyBracket)) {
    std::optional<Token> comma =
        parseOrTryRecover<&Parser::ParseToken<TK_Comma>>();
    if (!comma) {
      // TODO: Report error and recover
      return nullptr;
    }
    matchCase->setComma(std::move(*comma));
    cases.push_back(std::move(*matchCase));

    matchCase = MatchCase();
    if (!matchCase) {
      // Nothing to do here, reported on Expr
      return nullptr;
    }
  }
  cases.push_back(std::move(*matchCase));

  std::optional<Token> rKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_RKeyBracket>>();
  if (!rKeyBracket) {
    // TODO: Report error and recover
    return nullptr;
  }

  return std::make_unique<source::ExprMatch>(
      std::move(match), std::move(expr), std::move(*lKeyBracket),
      std::move(cases), std::move(*rKeyBracket));
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

  return std::make_unique<source::ExprTuple>(
      std::move(lparentheses), std::move(vals), std::move(*rparentheses));
}

std::optional<Token> Parser::missingCommaBetweenTupleElemsRecovery() {
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
    return nullopt;
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

std::optional<source::CompilationUnit>
tmplang::Parse(tmplang::Lexer &lex, raw_ostream &out, const SourceManager &sm) {
  return Parser(lex, out, sm).Start();
}
