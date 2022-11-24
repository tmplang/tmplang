#ifndef TMPLANG_PARSER_PARSER_H
#define TMPLANG_PARSER_PARSER_H

#include <llvm/ADT/Optional.h>
#include <tmplang/Lexer/Lexer.h>
#include <tmplang/Tree/Source/CompilationUnit.h>

namespace tmplang {

class SourceManager;

/// Simple grammar verifier parser. Given a lexer which already contains the
/// code returns wether the grammar can generate the input
llvm::Optional<source::CompilationUnit>
Parse(tmplang::Lexer &, llvm::raw_ostream &out, const SourceManager &);

} // namespace tmplang

#endif // TMPLANG_PARSER_PARSER_H
