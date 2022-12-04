#ifndef TMPLANG_LEXER_SOURCELOCATION_H
#define TMPLANG_LEXER_SOURCELOCATION_H

#include <cstdint>
#include <limits>

namespace tmplang {

/// Encoded source location, SourceManager is able to decode it. Must be kept
/// small. All nodes use this type so keep it small.
///
/// There are two special values:
///   <min>: means that the SourceLocation is invalid
///   <max>: means that the SourceLocation belongs to a recovery node
using SourceLocation = std::uint32_t;
static constexpr SourceLocation InvalidLoc =
    std::numeric_limits<SourceLocation>::min();
static constexpr SourceLocation RecoveryLoc =
    std::numeric_limits<SourceLocation>::max();

struct SourceLocationSpan {
  SourceLocation Start;
  SourceLocation End;

  bool operator==(const SourceLocationSpan &other) const = default;
};

} // namespace tmplang

#endif // TMPLANG_LEXER_SOURCELOCATION_H
