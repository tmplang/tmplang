#include <tmplang/Tree/Source/Decls.h>

using namespace tmplang::source;

/// fn foo: i32 a, f32 b -> i32 {}
/*static*/ FunctionDecl
FunctionDecl::Create(Token funcType, Token identifier, Token colon,
                     CommaSeparatedList<source::ParamDecl, 4> paramList,
                     ArrowAndType arrowAndType, Token lKeyBracket,
                     Token rKeyBracket) {
  return FunctionDecl(funcType, identifier, colon, std::move(paramList),
                      std::move(arrowAndType), lKeyBracket, rKeyBracket);
}
/// fn foo: i32 a, f32 b {}
/*static*/ FunctionDecl
FunctionDecl::Create(Token funcType, Token identifier, Token colon,
                     CommaSeparatedList<source::ParamDecl, 4> paramList,
                     Token lKeyBracket, Token rKeyBracket) {
  return FunctionDecl(funcType, identifier, colon, std::move(paramList),
                      llvm::None, lKeyBracket, rKeyBracket);
}
/// fn foo -> i32 {}
/*static*/ FunctionDecl FunctionDecl::Create(Token funcType, Token identifier,
                                             ArrowAndType arrowAndType,
                                             Token lKeyBracket,
                                             Token rKeyBracket) {
  return FunctionDecl(funcType, identifier, llvm::None, {},
                      std::move(arrowAndType), lKeyBracket, rKeyBracket);
}
/// fn foo {}
/*static*/ FunctionDecl FunctionDecl::Create(Token funcType, Token identifier,
                                             Token lKeyBracket,
                                             Token rKeyBracket) {
  return FunctionDecl(funcType, identifier, llvm::None, {}, llvm::None,
                      lKeyBracket, rKeyBracket);
}

FunctionDecl::FunctionDecl(Token funcType, Token identifier,
                           llvm::Optional<Token> colon,
                           CommaSeparatedList<source::ParamDecl, 4> paramList,
                           llvm::Optional<ArrowAndType> arrowAndType,
                           Token lKeyBracket, Token rKeyBracket)
    : Decl(Kind::FuncDecl), FuncType(funcType), Identifier(identifier),
      Colon(colon), ParamList(std::move(paramList)),
      OptArrowAndType(std::move(arrowAndType)), LKeyBracket(lKeyBracket),
      RKeyBracket(rKeyBracket) {}
