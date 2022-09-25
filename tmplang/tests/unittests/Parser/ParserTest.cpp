#include "tmplang/AST/ASTContext.h"
#include <Testing.h>

#include <tmplang/Parser/Parser.h>

using namespace tmplang;

TEST(ParserTest, Valid) {
  ASTContext ctx;
  std::array<llvm::StringLiteral, 10> tests = {
      "",
      "proc foo: Type a -> void {}",
      "proc foo: Type a, Type b -> void {}",
      "proc foo: Type a, Type b {}",
      "proc foo -> void {}",
      "proc foo {}",
      "fn foo: Type a -> void {}",
      "fn foo: Type a, Type b -> void {}",
      "fn foo -> void {}",
      "fn foo {}"};

  for (const llvm::StringLiteral &code : tests) {
    Lexer lex(code);
    EXPECT_TRUE(Parse(lex, ctx));
  }
}

TEST(ParserTest, Invalid) {
  ASTContext ctx;
  std::array<llvm::StringLiteral, 9> tests = {
      ":",
      "foo: Type a -> void {}",
      "proc : Type a -> void {}",
      "proc foo: Type a void {}",
      "proc foo Type a -> void {}",
      "proc foo Type a {}",
      "proc foo: Type a -> void }",
      "proc foo: -> void }",
      "proc foo: Type a -> void { {",
  };

  for (const llvm::StringLiteral &code : tests) {
    Lexer lex(code);
    EXPECT_FALSE(Parse(lex, ctx));
  }
}
