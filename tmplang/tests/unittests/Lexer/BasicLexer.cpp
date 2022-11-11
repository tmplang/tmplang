#include <Testing.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLArrayExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Lexer/Lexer.h>

#include <string>

using namespace tmplang;

static std::vector<Token> Lex(llvm::StringRef input) {
  Lexer lexer(input);
  Token tok = lexer.getCurrentToken();
  std::vector<Token> result = {tok};

  while (tok.Kind != TK_EOF && tok.Kind != TK_Unknown) {
    tok = lexer.next();
    result.push_back(tok);
  }

  return result;
}

static std::string Toks(llvm::ArrayRef<Token> resultTokens) {
  std::string result;
  llvm::raw_string_ostream o(result);
  bool first = true;
  for (Token tok : resultTokens) {
    if (!first) {
      o << ", ";
    }
    first = false;
    o << tok;
  }
  return o.str();
}

static void CheckResult(llvm::ArrayRef<Token> targetTokens,
                        llvm::ArrayRef<Token> resultTokens) {
  SCOPED_TRACE("Result tokens: {" + Toks(resultTokens) + "}\n");
  ASSERT_EQ(resultTokens.size(), targetTokens.size());
  for (size_t i = 0; i < resultTokens.size(); ++i) {
    SCOPED_TRACE("Token at index " + std::to_string(i));
    ASSERT_EQ(resultTokens[i], targetTokens[i]);
  }
}

static void TestLexer(llvm::StringRef input,
                      llvm::ArrayRef<Token> targetTokens) {
  SCOPED_TRACE("Input: " + input.str());
  CheckResult(targetTokens, Lex(input));
}

TEST(BasicLexer, Empty) { TestLexer("", {{TK_EOF, {1, 1}, {1, 1}}}); }

TEST(BasicLexer, EmptyFn) {
  TestLexer("fn foobar { }", {{TK_FnType, {1, 1}, {1, 2}},
                              {"foobar", {1, 4}, {1, 9}},
                              {TK_LKeyBracket, {1, 11}, {1, 11}},
                              {TK_RKeyBracket, {1, 13}, {1, 13}},
                              {TK_EOF, {1, 14}, {1, 14}}});
}

TEST(BasicLexer, LiteralError) {
  TestLexer("42abc", {{TK_Unknown, {1, 1}, {1, 1}}});
}

TEST(BasicLexer, Joined) {
  Token tokens[] = {
      {"foo", {1, 1}, {1, 3}},      {TK_Comma, {1, 4}, {1, 4}},
      {"bar", {1, 5}, {1, 7}},      {TK_LKeyBracket, {1, 8}, {1, 8}},
      {TK_FnType, {1, 9}, {1, 10}}, {TK_EOF, {1, 11}, {1, 11}},
  };

  TestLexer("foo,bar{fn", tokens);
}

TEST(BasicLexer, FullFunction) {
  Token tokens[] = {
      {TK_FnType, {1, 1}, {1, 2}},
      {"func", {1, 4}, {1, 7}},
      {TK_Colon, {1, 8}, {1, 8}},
      {"i32", {1, 10}, {1, 12}},
      {"a", {1, 14}, {1, 14}},
      {TK_Comma, {1, 15}, {1, 15}},
      {"f32", {1, 17}, {1, 19}},
      {"b", {1, 21}, {1, 21}},
      {TK_LKeyBracket, {1, 23}, {1, 23}},
      {TK_RKeyBracket, {1, 24}, {1, 24}},
      {TK_EOF, {1, 25}, {1, 25}},
  };

  TestLexer("fn func: i32 a, f32 b {}", tokens);
}

TEST(BasicLexer, Emoji) {
  TestLexer("🙂🙂🙂", {{TK_Unknown, {1, 1}, {1, 1}}});
  TestLexer("🙂a 🍟123 🥰x", {{TK_Unknown, {1, 1}, {1, 1}}});
}

TEST(BasicLexer, UTF8) {
  TestLexer("这是一个演示", {{TK_Unknown, {1, 1}, {1, 1}}});
  // Right-to-left text is a bit mindblowing... I guess this should still be
  // allowed? Or only Right-to-left in the identifiers/words themselves? (not
  // the whole text)
  TestLexer("fn الألم; بعض {:بعض", {
                                       {TK_FnType, {1, 1}, {1, 2}},
                                       {TK_Unknown, {1, 4}, {1, 4}},
                                   });
}

TEST(BasicLexer, MultiLine) {
  TestLexer("\tfn foo: a Chr -> Int {\n\n\t}",
            {
                {TK_FnType, {1, 2}, {1, 3}},
                {"foo", {1, 5}, {1, 7}},
                {TK_Colon, {1, 8}, {1, 8}},
                {"a", {1, 10}, {1, 10}},
                {"Chr", {1, 12}, {1, 14}},
                {TK_RArrow, {1, 16}, {1, 17}},
                {"Int", {1, 19}, {1, 21}},
                {TK_LKeyBracket, {1, 23}, {1, 23}},
                {TK_RKeyBracket, {3, 2}, {3, 2}},
                {TK_EOF, {3, 3}, {3, 3}},
            });
}

TEST(BasicLexer, Comment) {
  TestLexer("// fn func: Typ a, Typ b {}", {{TK_EOF, {1, 28}, {1, 28}}});
}

TEST(BasicLexer, InvalidComment) {
  TestLexer("/ fn func: Typ a, Typ b {}", {{TK_Unknown, {1, 1}, {1, 1}}});
}

TEST(BasicLexer, MultiLineWithComment) {
  llvm::StringLiteral code = R"(
//This functions does basically nothing
fn foo: a Chr -> Int {
  // Implement this function, do something cool
}//)";

  TestLexer(code, {{TK_FnType, {3, 1}, {3, 2}},
                   {"foo", {3, 4}, {3, 6}},
                   {TK_Colon, {3, 7}, {3, 7}},
                   {"a", {3, 9}, {3, 9}},
                   {"Chr", {3, 11}, {3, 13}},
                   {TK_RArrow, {3, 15}, {3, 16}},
                   {"Int", {3, 18}, {3, 20}},
                   {TK_LKeyBracket, {3, 22}, {3, 22}},
                   {TK_RKeyBracket, {5, 1}, {5, 1}},
                   {TK_EOF, {5, 4}, {5, 4}}});
}

TEST(BasicLexer, Parentheses) {
  TestLexer("fn foo -> (i32, i32)", {
                                        {TK_FnType, {1, 1}, {1, 2}},
                                        {"foo", {1, 4}, {1, 6}},
                                        {TK_RArrow, {1, 8}, {1, 9}},
                                        {TK_LParentheses, {1, 11}, {1, 11}},
                                        {"i32", {1, 12}, {1, 14}},
                                        {TK_Comma, {1, 15}, {1, 15}},
                                        {"i32", {1, 17}, {1, 19}},
                                        {TK_RParentheses, {1, 20}, {1, 20}},
                                        {TK_EOF, {1, 21}, {1, 21}},
                                    });
}
