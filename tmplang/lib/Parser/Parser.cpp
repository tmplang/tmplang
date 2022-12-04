#include "tmplang/Lexer/Token.h"
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
      : Lex(lex), Out(out), SM(sm) {
    // These two "consume" calls will initialize the current and next token
    consume();
    consume();
  }

  llvm::Optional<source::CompilationUnit> Start();

private:
  llvm::Optional<source::FunctionDecl> FunctionDefinition();
  llvm::Optional<source::FunctionDecl> ArrowAndEndOfSubprogramFactored(
      Token funcType, Token id, llvm::Optional<Token> colon = llvm::None,
      llvm::SmallVector<source::ParamDecl, 4> paramList = {});
  llvm::Optional<Token> missingSubprogramTypeRecovery();
  llvm::Optional<Token> missingSubprogramIdRecovery();
  llvm::Optional<Token> missingReturnTypeArrowRecovery();

  llvm::Optional<llvm::SmallVector<source::ParamDecl, 4>> ParamList();
  llvm::Optional<Token> missingCommaParamSepRecovery();

  llvm::Optional<source::ParamDecl> Param();
  llvm::Optional<Token> missingVariableOnParamRecovery();

  llvm::Optional<LexicalScope> Block();
  llvm::Optional<Token> missingLeftKeyBracketRecovery();
  llvm::Optional<Token> missingRightKeyBracketRecovery();

  llvm::Optional<Token> Identifier();

  // Types...
  source::RAIIType Type();
  source::RAIIType NamedType();
  source::RAIIType TupleType();
  llvm::Optional<Token> missingCommaBetweenTypesRecovery();
  llvm::Optional<Token> missingRightParenthesesOfTupleRecovery();

  // Simple token parsing. This function is useful in conjuntion with
  // parseOrRecover so it can recieve the address
  template <TokenKind... Kinds> llvm::Optional<Token> ParseToken() {
    if (tk().isOneOf(Kinds...)) {
      return consume();
    }
    return llvm::None;
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
  Token getRecoveryToken(llvm::StringRef id);

  template <llvm::Optional<Token> (Parser::*parsingFunc)(),
            llvm::Optional<Token> (Parser::*recoveryFunc)() = nullptr>
  llvm::Optional<Token> parseOrTryRecover(bool emitUnexpectedTokenDiag = true) {
    // Try the normal path, parse as it is expected
    if (auto parsedToken = (this->*parsingFunc)()) {
      return *parsedToken;
    }

    // If there is no recovery function, do not recover
    if (recoveryFunc == nullptr) {
      return llvm::None;
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

    return llvm::None;
  }

private:
  Lexer &Lex;
  llvm::raw_ostream &Out;
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
llvm::Optional<source::CompilationUnit> Parser::Start() {
  std::vector<source::FunctionDecl> functionDeclarations;
  while (true) {
    if (tk().is(TokenKind::TK_EOF)) {
      return source::CompilationUnit(std::move(functionDeclarations),
                                     ParserState.NumberOfRecoveriesPerformed);
    }

    auto func = FunctionDefinition();
    if (!func) {
      // Nothing to do, already reported
      return llvm::None;
    }

    functionDeclarations.push_back(std::move(*func));
  }

  return source::CompilationUnit(std::move(functionDeclarations),
                                 ParserState.NumberOfRecoveriesPerformed);
}

llvm::Optional<source::FunctionDecl> Parser::ArrowAndEndOfSubprogramFactored(
    Token funcType, Token id, llvm::Optional<Token> colon,
    llvm::SmallVector<source::ParamDecl, 4> paramList) {
  if (auto arrow = parseOrTryRecover<&Parser::ParseToken<TK_RArrow>,
                                     &Parser::missingReturnTypeArrowRecovery>(
          /*emitUnexpectedTokenDiag*/ false)) {
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

    return source::FunctionDecl(
        funcType, id, block->LKeyBracket, block->RKeyBracket, colon,
        std::move(paramList),
        source::FunctionDecl::ArrowAndType{*arrow, std::move(returnType)});
  }

  // [2]
  auto block = Block();
  if (!block) {
    // Nothing to report here, reported on Block
    return llvm::None;
  }

  return source::FunctionDecl(funcType, id, block->LKeyBracket,
                              block->RKeyBracket, colon, std::move(paramList));
}

/// Function_type = "proc" | "fn";

/// Function_Definition =
///  [1] | Function_Type, Identifier, ":", Param_List, "->", Type, Block
///  [2] | Function_Type, Identifier, ":", Param_List, Block
///  [3] | Function_Type, Identifier, "->", Type, Block
///  [4] | Function_Type, Identifier, Block;
llvm::Optional<source::FunctionDecl> Parser::FunctionDefinition() {
  auto funcType = parseOrTryRecover<&Parser::ParseToken<TK_FnType, TK_ProcType>,
                                    &Parser::missingSubprogramTypeRecovery>();
  if (!funcType) {
    // Since this is the start of top level declaration, lets consume the
    // unknowns until we find something we understand
    if (tk().is(tmplang::TK_Unknown)) {
      consume();
      return FunctionDefinition();
    }
    return llvm::None;
  }

  auto id = parseOrTryRecover<&Parser::Identifier,
                              &Parser::missingSubprogramIdRecovery>();
  if (!id) {
    return llvm::None;
  }

  // [1] && [2]
  if (tk().is(TK_Colon)) {
    auto colon = consume();

    auto paramList = ParamList();
    if (!paramList) {
      // Nothing to report here, reported on ParamList
      return llvm::None;
    }

    return ArrowAndEndOfSubprogramFactored(*funcType, *id, colon,
                                           std::move(*paramList));
  }

  return ArrowAndEndOfSubprogramFactored(*funcType, *id);
}

llvm::Optional<Token> Parser::missingSubprogramTypeRecovery() {
  const bool potentialStartOfSubprogram =
      tk().is(TK_Identifier) &&
      nextTk().isOneOf(TK_Colon, TK_RArrow, TK_LKeyBracket);
  if (!potentialStartOfSubprogram) {
    return llvm::None;
  }

  Diagnostic(DiagId::err_missing_subprogram_class, tk().getSpan(),
             InsertTextAtHint(tk().getSpan().Start,
                              {ToString(TK_FnType), ToString(TK_ProcType)}, ""))
      .print(Out, SM);

  return getRecoveryToken(TK_FnType);
}

llvm::Optional<Token> Parser::missingSubprogramIdRecovery() {
  const bool missingId = prevTk().isOneOf(TK_FnType, TK_ProcType) &&
                         tk().isOneOf(TK_Colon, TK_RArrow, TK_LKeyBracket);
  if (!missingId) {
    return llvm::None;
  }

  constexpr llvm::StringLiteral placeHolder = "<subprogram_identifier>";

  Diagnostic(DiagId::err_missing_subprogram_id, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), placeHolder))
      .print(Out, SM);

  return getRecoveryToken(placeHolder);
}

llvm::Optional<Token> Parser::missingReturnTypeArrowRecovery() {
  const bool missingArrow =
      tk().isOneOf(/*firstTokensOfType*/ TK_Identifier, TK_LParentheses);
  if (!missingArrow) {
    return llvm::None;
  }

  Diagnostic(DiagId::err_missing_arrow, prevTk().getSpan(),
             InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_RArrow)))
      .print(Out, SM);

  return getRecoveryToken(TK_RArrow);
}

/// Param_List = Param (",", Param)*;
llvm::Optional<llvm::SmallVector<source::ParamDecl, 4>> Parser::ParamList() {
  llvm::SmallVector<source::ParamDecl, 4> paramList;

  auto firstParam = Param();
  if (!firstParam) {
    // Nothing to report here, reported on Param
    return llvm::None;
  }
  paramList.push_back(std::move(*firstParam));

  while (auto comma = parseOrTryRecover<&Parser::ParseToken<TK_Comma>,
                                        &Parser::missingCommaParamSepRecovery>(
             /*emitUnexpectedTokenDiag*/ false)) {
    paramList.back().setComma(*comma);

    auto param = Param();
    if (!param) {
      // Nothing to report here, reported on Param
      return llvm::None;
    }

    paramList.push_back(std::move(*param));
  }

  return paramList;
}

llvm::Optional<Token> Parser::missingCommaParamSepRecovery() {
  // fn foo: i32 var i32 var ...
  //                ^___ comma here

  // fn foo: i32 var () var ...
  // fn foo: i32 var (var) var ...
  //   do not emit in the above cases, the can get confusing with missing arrow
  const bool missingComma = prevTk().is(TK_Identifier) &&
                            tk().is(TK_Identifier) &&
                            nextTk().is(TK_Identifier);
  if (!missingComma) {
    return llvm::None;
  }

  Diagnostic(DiagId::err_missing_comma, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  return getRecoveryToken(TK_Comma);
}

/// Param = Type Identifier;
llvm::Optional<source::ParamDecl> Parser::Param() {
  auto type = Type();
  if (!type) {
    // Nothing to report here, reported on Type
    return llvm::None;
  }

  auto paramId = parseOrTryRecover<&Parser::Identifier,
                                   &Parser::missingVariableOnParamRecovery>();
  if (!paramId) {
    return llvm::None;
  }

  return source::ParamDecl(std::move(type), *paramId);
}

llvm::Optional<Token> Parser::missingVariableOnParamRecovery() {
  constexpr llvm::StringLiteral paramId = "<parameter_id>";

  Diagnostic(DiagId::err_missing_variable_identifier_after_type, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), paramId))
      .print(Out, SM);

  return getRecoveryToken(paramId);
}

/// Block = "{" "}";
llvm::Optional<LexicalScope> Parser::Block() {
  auto lKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_LKeyBracket>,
                        &Parser::missingLeftKeyBracketRecovery>();
  if (!lKeyBracket) {
    return llvm::None;
  }

  auto rKeyBracket =
      parseOrTryRecover<&Parser::ParseToken<TK_RKeyBracket>,
                        &Parser::missingRightKeyBracketRecovery>();
  if (!rKeyBracket) {
    return llvm::None;
  }

  return LexicalScope{*lKeyBracket, *rKeyBracket};
}

llvm::Optional<Token> Parser::missingLeftKeyBracketRecovery() {
  Diagnostic(
      DiagId::err_missing_left_key_brace, prevTk().getSpan(),
      InsertTextAtHint(prevTk().getSpan().End + 1, ToString(TK_LKeyBracket)))
      .print(Out, SM);

  return getRecoveryToken(TK_LKeyBracket);
}

llvm::Optional<Token> Parser::missingRightKeyBracketRecovery() {
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

  llvm::SmallVector<source::RAIIType, 4> types;
  llvm::SmallVector<Token, 3> commas;

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

llvm::Optional<Token> Parser::missingCommaBetweenTypesRecovery() {
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
    return llvm::None;
  }

  Diagnostic(DiagId::err_missing_comma_prior_tuple_param, tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_Comma)))
      .print(Out, SM);

  return getRecoveryToken(TK_Comma);
}

llvm::Optional<Token> Parser::missingRightParenthesesOfTupleRecovery() {
  Diagnostic(DiagId::err_missing_right_parenthesis_closing_tuple,
             tk().getSpan(),
             InsertTextAtHint(getStartCurrToken(), ToString(TK_RParentheses)))
      .print(Out, SM);
  return getRecoveryToken(TK_RParentheses);
}

/// Identifier = [a-zA-Z][a-zA-Z0-9]*;
llvm::Optional<Token> Parser::Identifier() {
  return tk().is(TK_Identifier) ? consume() : llvm::Optional<Token>{};
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

Token Parser::getRecoveryToken(llvm::StringRef id) {
  ParserState.NumberOfRecoveriesPerformed++;
  return Token(id, RecoveryLoc, RecoveryLoc, /*isRecovery=*/true);
}

llvm::Optional<source::CompilationUnit>
tmplang::Parse(tmplang::Lexer &lex, llvm::raw_ostream &out,
               const SourceManager &sm) {
  return Parser(lex, out, sm).Start();
}
