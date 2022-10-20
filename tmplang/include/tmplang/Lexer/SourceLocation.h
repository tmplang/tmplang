#ifndef TMPLANG_LEXER_SOURCELOCATION_H
#define TMPLANG_LEXER_SOURCELOCATION_H

namespace tmplang {

/// Basic struct to hold positions on the source
struct SourceLocation {
  SourceLocation() = default;
  SourceLocation(unsigned line, unsigned column) : Line(line), Column(column) {}

  bool operator==(const SourceLocation &other) const = default;

  unsigned Line = 0;
  unsigned Column = 0;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_SOURCELOCATION_H
