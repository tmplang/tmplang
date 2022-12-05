#ifndef LIB_TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H
#define LIB_TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/ADT/LLVM.h>

namespace tmplang {

[[maybe_unused]] static constexpr raw_ostream::Colors RED = raw_ostream::RED;
[[maybe_unused]] static constexpr raw_ostream::Colors WHITE =
    raw_ostream::WHITE;
[[maybe_unused]] static constexpr raw_ostream::Colors YELLOW =
    raw_ostream::YELLOW;
[[maybe_unused]] static constexpr raw_ostream::Colors GREEN =
    raw_ostream::GREEN;
[[maybe_unused]] static constexpr raw_ostream::Colors CYAN = raw_ostream::CYAN;
[[maybe_unused]] static constexpr raw_ostream::Colors BLACK =
    raw_ostream::BLACK;
[[maybe_unused]] static constexpr raw_ostream::Colors BLUE = raw_ostream::BLUE;
[[maybe_unused]] static constexpr raw_ostream::Colors MAGENTA =
    raw_ostream::MAGENTA;

constexpr static char caret = '^';
constexpr static char tilde = '~';
constexpr static char space = ' ';

SmallString<80> GetSubscriptLine(unsigned start, unsigned size);

void PrintContextLine(raw_ostream &out, StringRef lhs,
                      const Twine &rhs = Twine(""),
                      raw_ostream::Colors rhsColor = WHITE);

} // namespace tmplang

#endif // LIB_TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H
