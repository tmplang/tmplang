#ifndef TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H
#define TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <tmplang/Lexer/Token.h>

namespace tmplang::source {

template <typename T, unsigned N> struct CommaSeparatedList {
  CommaSeparatedList() = default;
  CommaSeparatedList(SmallVectorImpl<T> &&elems,
                     SmallVectorImpl<Token> &&commas)
      : Elems(std::move(elems)) {
    llvm::move(std::move(commas), std::back_inserter(Commas));

    assert((Elems.empty() && Commas.empty()) ||
           (Elems.size() == Commas.size() + 1));
  }

  SmallVector<T, N> Elems;
  SmallVector<SpecificToken<TK_Comma>, N - 1> Commas;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H
