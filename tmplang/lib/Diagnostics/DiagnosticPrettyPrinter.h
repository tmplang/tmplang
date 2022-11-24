#ifndef LIB_TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H
#define LIB_TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>

namespace tmplang {

[[maybe_unused]] static constexpr llvm::raw_ostream::Colors RED =
    llvm::raw_ostream::RED;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors WHITE =
    llvm::raw_ostream::WHITE;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors YELLOW =
    llvm::raw_ostream::YELLOW;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors GREEN =
    llvm::raw_ostream::GREEN;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors CYAN =
    llvm::raw_ostream::CYAN;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors BLACK =
    llvm::raw_ostream::BLACK;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors BLUE =
    llvm::raw_ostream::BLUE;
[[maybe_unused]] static constexpr llvm::raw_ostream::Colors MAGENTA =
    llvm::raw_ostream::MAGENTA;

constexpr static char caret = '^';
constexpr static char tilde = '~';
constexpr static char space = ' ';

llvm::SmallString<80> GetSubscriptLine(unsigned start, unsigned size);

void PrintContextLine(llvm::raw_ostream &out, llvm::StringRef lhs,
                      const llvm::Twine &rhs = llvm::Twine(""),
                      llvm::raw_ostream::Colors rhsColor = WHITE);

} // namespace tmplang

#endif // LIB_TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H
