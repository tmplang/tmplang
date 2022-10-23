#include <Testing.h>

#include <tmplang/Parser/Parser.h>
#include <tmplang/Tree/HIR/HIRBuilder.h>

using namespace tmplang;

TEST(HIRTest, Invalid) {
  // Neither "Type", nor "void" are valid types
  llvm::StringLiteral tests[] = {
      "proc foo: Type a -> void {}", "proc foo: Type a, Type b -> void {}",
      "proc foo: Type a, Type b {}", "proc foo -> void {}",
      "fn foo: Type a -> void {}",   "fn foo: Type a, Type b -> void {}",
      "fn foo -> void {}",
  };

  for (const llvm::StringLiteral &code : tests) {
    Lexer lex(code);
    auto srcCompUnit = Parse(lex);
    ASSERT_TRUE(srcCompUnit);
    hir::HIRContext ctx;
    auto hirCompUnit = hir::buildHIR(*srcCompUnit, ctx);
    EXPECT_FALSE(hirCompUnit);
  }
}
