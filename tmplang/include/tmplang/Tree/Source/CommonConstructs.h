#ifndef TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H
#define TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H

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

/// Some nodes have a trailing optional comma, this struct servers as
/// factorization
class TrailingOptComma {
public:
  const std::optional<SpecificToken<TK_Comma>> &getComma() const { return Comma; }
  void setComma(SpecificToken<TK_Comma> comma) { Comma = std::move(comma); }

private:
  std::optional<SpecificToken<TK_Comma>> Comma;
};

/// Wrapper arround std::vector to guarantee a desired mininum size using
/// runtime asserts
template <typename T, unsigned MinSize> struct MinElementList {
  using InternalList_t = std::vector<T>;

  MinElementList() { assert(MinSize == 0 && "Minimun size is 0"); }

  MinElementList(InternalList_t items) : Items(std::move(items)) {
    assert(Items.size() >= MinSize && "Minimun size is MinSize");
  }

  InternalList_t Items;
};

template <typename T> using VariadicList = MinElementList<T, 0>;
template <typename T> using OneElementOrMoreList = MinElementList<T, 1>;

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H
