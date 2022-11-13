#ifndef TMPLANG_LEXER_SOURCELOCATION_H
#define TMPLANG_LEXER_SOURCELOCATION_H

#include <cstdint>

namespace tmplang {

/// Encoded source location, SourceManager is able to decode it. Must be ketp
/// small. All nodes use this type so keep it small
using SourceLocation = std::uint32_t;

struct SourceLocationSpan {
  SourceLocation Start;
  SourceLocation End;

  bool operator==(const SourceLocationSpan  &other) const = default;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_SOURCELOCATION_H
