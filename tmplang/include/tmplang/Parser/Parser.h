#ifndef TMPLANG_PARSER_PARSER_H
#define TMPLANG_PARSER_PARSER_H

#include <llvm/ADT/Optional.h>
#include <tmplang/AST/CompilationUnit.h>
#include <tmplang/Lexer/Lexer.h>

namespace tmplang {

class ASTContext;

/// Simple grammar verifier parser. Given a lexer which already contains the
/// code returns wether the grammar can generate the input
llvm::Optional<CompilationUnit> Parse(tmplang::Lexer &lex, ASTContext &ct);

} // namespace tmplang

#endif // TMPLANG_PARSER_PARSER_H
