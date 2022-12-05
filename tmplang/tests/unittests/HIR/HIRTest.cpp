#include <Testing.h>

#include "../common/Parser.h"

using namespace tmplang;

TEST(HIRTest, Invalid) {
  // Neither "Type", nor "void" are valid types
  StringLiteral tests[] = {
      "proc foo: Type a -> void {}", "proc foo: Type a, Type b -> void {}",
      "proc foo: Type a, Type b {}", "proc foo -> void {}",
      "fn foo: Type a -> void {}",   "fn foo: Type a, Type b -> void {}",
      "fn foo -> void {}",
  };

  for (const StringLiteral &code : tests) {
    auto srcCompUnit = CleanParse(code);
    ASSERT_TRUE(srcCompUnit);
    hir::HIRContext ctx;
    auto hirCompUnit = hir::buildHIR(*srcCompUnit, ctx);
    EXPECT_FALSE(hirCompUnit);
  }
}
