#include <Testing.h>

#include "../common/Parser.h"

using namespace tmplang;

TEST(ParserTest, Valid) {
  std::array<StringLiteral, 10> tests = {"",
                                         "proc foo: Type a -> void {}",
                                         "proc foo: Type a, Type b -> void {}",
                                         "proc foo: Type a, Type b {}",
                                         "proc foo -> void {}",
                                         "proc foo {}",
                                         "fn foo: Type a -> void {}",
                                         "fn foo: Type a, Type b -> void {}",
                                         "fn foo -> void {}",
                                         "fn foo {}"};

  for (const StringLiteral &code : tests) {
    EXPECT_TRUE(CleanParse(code));
  }
}

TEST(ParserTest, Invalid) {
  std::array<StringLiteral, 9> tests = {
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

  for (const StringLiteral &code : tests) {
    std::optional<source::CompilationUnit> compUnit = CleanParse(code);
    EXPECT_TRUE(!compUnit || compUnit->didRecoverFromAnError());
  }
}
