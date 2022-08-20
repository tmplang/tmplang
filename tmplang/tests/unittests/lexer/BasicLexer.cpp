#include <Testing.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLArrayExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Lexer/Lexer.h>

#include <string>

using namespace tmplang;

static std::vector<Token> Lex(llvm::StringRef input) {
  std::vector<Token> result;
  Lexer lexer(input);

  Token tok{TK_Unknown};
  do {
    tok = lexer.next();
    result.push_back(tok);
  } while (tok.Kind != TK_EOF && tok.Kind != TK_Unknown);

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

TEST(BasicLexer, EmptyFn) {
  TestLexer("fn foobar { }", {{TK_FnType},
                              {TK_Identifier},
                              {TK_LKeyBracket},
                              {TK_RKeyBracket},
                              {TK_EOF}});
}

TEST(BasicLexer, LiteralError) { TestLexer("42abc", {{TK_Unknown}}); }

TEST(BasicLexer, Joined) {
  TestLexer("foo,bar{fn", {{TK_Identifier},
                           {TK_Comma},
                           {TK_Identifier},
                           {TK_LKeyBracket},
                           {TK_FnType},
                           {TK_EOF}});
}

TEST(BasicLexer, Emoji) {
  TestLexer("🙂🙂🙂", {{TK_Unknown}});
  TestLexer("🙂a 🍟123 🥰x", {{TK_Unknown}});
}

TEST(BasicLexer, UTF8) {
  TestLexer("这是一个演示", {{TK_Unknown}});
  // Right-to-left text is a bit mindblowing... I guess this should still be
  // allowed? Or only Right-to-left in the identifiers/words themselves? (not
  // the whole text)
  TestLexer("fn الألم; بعض {:بعض", {
                                       {TK_FnType},
                                       {TK_Unknown},
                                   });
}
