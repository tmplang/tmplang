#ifndef TMPLANG_ADT_PRINTUTILS_H
#define TMPLANG_ADT_PRINTUTILS_H

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/ADT/LLVM.h>

namespace tmplang {

template <typename Container, typename Printer>
void printInterleaved(const Container &container, Printer printer,
                      raw_ostream &os, StringRef sep = ", ") {
  bool first = true;
  for (auto &element : container) {
    if (!first) {
      os << sep;
    }

    first = false;
    printer(element);
  }
}

template <typename Container>
void printInterleaved(const Container &container, raw_ostream &os,
                      StringRef sep = ", ") {
  printInterleaved(
      container, [&os](auto &e) { os << e; }, os, sep);
}

} // namespace tmplang

#endif // TMPLANG_ADT_PRINTUTILS_H
