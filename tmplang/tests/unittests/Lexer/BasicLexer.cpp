#include <Testing.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Lexer/Lexer.h>
#include <tmplang/Support/FileManager.h>
#include <tmplang/Support/SourceManager.h>

#include <string>

using namespace tmplang;

static std::vector<Token> Lex(StringRef input) {
  Lexer lexer(input);
  Token tok = lexer.next();
  std::vector<Token> result = {tok};

  while (!tok.isOneOf(TK_EOF, TK_Unknown)) {
    tok = lexer.next();
    result.push_back(tok);
  }

  return result;
}

static std::string Toks(ArrayRef<Token> resultTokens, const SourceManager &sm) {
  std::string result;
  raw_string_ostream o(result);
  llvm::interleaveComma(resultTokens, o, [&](Token tok) { tok.print(o, sm); });
  return o.str();
}

namespace {
/// Imitates a Token but with Line and Column, instead of encoded offset
struct MimicToken {
  MimicToken(StringRef id, TokenKind kind, LineAndColumn start,
             LineAndColumn end)
      : Lex(id), Kind(kind), Start(start), End(end) {}
  MimicToken(TokenKind kind, LineAndColumn start, LineAndColumn end)
      : Lex(ToString(kind)), Kind(kind), Start(start), End(end) {}

  StringRef Lex;
  TokenKind Kind;
  LineAndColumn Start;
  LineAndColumn End;
};
} // namespace

static void CheckResult(ArrayRef<MimicToken> targetTokens,
                        ArrayRef<Token> resultTokens, const SourceManager &sm) {
  SCOPED_TRACE("Result tokens: {" + Toks(resultTokens, sm) + "}\n");
  ASSERT_EQ(resultTokens.size(), targetTokens.size());

  for (size_t i = 0; i < resultTokens.size(); ++i) {
    SCOPED_TRACE("Token at index " + std::to_string(i));

    const LineAndColumn begin =
        sm.getLineAndColumn(resultTokens[i].getSpan().Start);
    const LineAndColumn end =
        sm.getLineAndColumn(resultTokens[i].getSpan().End);

    EXPECT_EQ(begin, targetTokens[i].Start);
    EXPECT_EQ(end, targetTokens[i].End);
    EXPECT_TRUE(resultTokens[i].is(targetTokens[i].Kind));
    if (resultTokens[i].is(TK_Identifier)) {
      EXPECT_EQ(resultTokens[i].getLexeme(), targetTokens[i].Lex);
    }
  }
}

static void TestLexer(StringRef input, ArrayRef<MimicToken> targetTokens) {
  auto inMemoryFileSystem = std::make_unique<llvm::vfs::InMemoryFileSystem>();

  const char *fileName = "./test";
  inMemoryFileSystem->addFile(fileName, 0,
                              llvm::MemoryBuffer::getMemBuffer(input));

  FileManager fm(std::move(inMemoryFileSystem));
  const TargetFileEntry *tfe = fm.findOrOpenTargetFile(fileName);
  assert(tfe);

  const SourceManager sm(*tfe);

  SCOPED_TRACE("Input: " + input.str());
  CheckResult(targetTokens, Lex(input), sm);
}

TEST(BasicLexer, Empty) { TestLexer("", {{TK_EOF, {1, 1}, {1, 1}}}); }

TEST(BasicLexer, EmptyFn) {
  TestLexer("fn foobar { }", {{TK_FnType, {1, 1}, {1, 2}},
                              {"foobar", TK_Identifier, {1, 4}, {1, 9}},
                              {TK_LKeyBracket, {1, 11}, {1, 11}},
                              {TK_RKeyBracket, {1, 13}, {1, 13}},
                              {TK_EOF, {1, 14}, {1, 14}}});
}

TEST(BasicLexer, Joined) {
  MimicToken tokens[] = {
      {"foo", TK_Identifier, {1, 1}, {1, 3}}, {TK_Comma, {1, 4}, {1, 4}},
      {"bar", TK_Identifier, {1, 5}, {1, 7}}, {TK_LKeyBracket, {1, 8}, {1, 8}},
      {TK_FnType, {1, 9}, {1, 10}},           {TK_EOF, {1, 11}, {1, 11}},
  };

  TestLexer("foo,bar{fn", tokens);
}

TEST(BasicLexer, FullFunction) {
  MimicToken tokens[] = {
      {TK_FnType, {1, 1}, {1, 2}},
      {"func", TK_Identifier, {1, 4}, {1, 7}},
      {TK_Colon, {1, 8}, {1, 8}},
      {"i32", TK_Identifier, {1, 10}, {1, 12}},
      {"a", TK_Identifier, {1, 14}, {1, 14}},
      {TK_Comma, {1, 15}, {1, 15}},
      {"f32", TK_Identifier, {1, 17}, {1, 19}},
      {"b", TK_Identifier, {1, 21}, {1, 21}},
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
                {"foo", TK_Identifier, {1, 5}, {1, 7}},
                {TK_Colon, {1, 8}, {1, 8}},
                {"a", TK_Identifier, {1, 10}, {1, 10}},
                {"Chr", TK_Identifier, {1, 12}, {1, 14}},
                {TK_RArrow, {1, 16}, {1, 17}},
                {"Int", TK_Identifier, {1, 19}, {1, 21}},
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
  StringLiteral code = R"(
//This functions does basically nothing
fn foo: a Chr -> Int {
  // Implement this function, do something cool
}//)";

  TestLexer(code, {{TK_FnType, {3, 1}, {3, 2}},
                   {"foo", TK_Identifier, {3, 4}, {3, 6}},
                   {TK_Colon, {3, 7}, {3, 7}},
                   {"a", TK_Identifier, {3, 9}, {3, 9}},
                   {"Chr", TK_Identifier, {3, 11}, {3, 13}},
                   {TK_RArrow, {3, 15}, {3, 16}},
                   {"Int", TK_Identifier, {3, 18}, {3, 20}},
                   {TK_LKeyBracket, {3, 22}, {3, 22}},
                   {TK_RKeyBracket, {5, 1}, {5, 1}},
                   {TK_EOF, {5, 4}, {5, 4}}});
}

TEST(BasicLexer, Parentheses) {
  TestLexer("fn foo -> (i32, i32)",
            {
                {TK_FnType, {1, 1}, {1, 2}},
                {"foo", TK_Identifier, {1, 4}, {1, 6}},
                {TK_RArrow, {1, 8}, {1, 9}},
                {TK_LParentheses, {1, 11}, {1, 11}},
                {"i32", TK_Identifier, {1, 12}, {1, 14}},
                {TK_Comma, {1, 15}, {1, 15}},
                {"i32", TK_Identifier, {1, 17}, {1, 19}},
                {TK_RParentheses, {1, 20}, {1, 20}},
                {TK_EOF, {1, 21}, {1, 21}},
            });
}

TEST(BasicLexer, Numbers) {
  TestLexer("1", {
                     {"1", TK_IntegerNumber, {1, 1}, {1, 1}},
                     {TK_EOF, {1, 2}, {1, 2}},
                 });
  TestLexer("1000", {
                        {"1000", TK_IntegerNumber, {1, 1}, {1, 4}},
                        {TK_EOF, {1, 5}, {1, 5}},
                    });
  TestLexer("1_000_000", {
                             {"1_000_000", TK_IntegerNumber, {1, 1}, {1, 9}},
                             {TK_EOF, {1, 10}, {1, 10}},
                         });
  TestLexer("1_____00", {
                            {"1_____00", TK_IntegerNumber, {1, 1}, {1, 8}},
                            {TK_EOF, {1, 9}, {1, 9}},
                        });
}

TEST(BasicLexer, InvalidNumber) {
  TestLexer("_1", {{TK_Unknown, {1, 1}, {1, 1}}});
}

TEST(BasicLexer, StartOfData) {
  TestLexer("data =", {
                          {TK_Data, {1, 1}, {1, 4}},
                          {TK_Eq, {1, 6}, {1, 6}},
                          {TK_EOF, {1, 7}, {1, 7}},
                      });
}
