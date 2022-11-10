#ifndef TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H
#define TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <tmplang/Lexer/Token.h>

namespace tmplang::source {

template <typename T, unsigned N> struct CommaSeparatedList {
  CommaSeparatedList() = default;
  CommaSeparatedList(llvm::SmallVectorImpl<T> &&elems,
                     llvm::SmallVectorImpl<Token> &&commas)
      : Elems(std::move(elems)), Commas(std::move(commas)) {
    assert((Elems.empty() && Commas.empty()) ||
           (Elems.size() == Commas.size() + 1));
    assert(llvm::all_of(Commas,
                       [](const Token &tk) { return tk.Kind == TK_Comma; }));
  }

  llvm::SmallVector<T, N> Elems;
  llvm::SmallVector<Token, N - 1> Commas;
};

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H
