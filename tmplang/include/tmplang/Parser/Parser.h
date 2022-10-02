#ifndef TMPLANG_PARSER_PARSER_H
#define TMPLANG_PARSER_PARSER_H

#include <tmplang/Lexer/Lexer.h>

namespace tmplang {

class HIRContext;

/// Simple grammar verifier parser. Given a lexer which already contains the code
/// returns wether the grammar can generate the input
bool Parse(tmplang::Lexer &lex);

} // namespace tmplang

#endif // TMPLANG_PARSER_PARSER_H
