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

} // namespace tmplang::source

#endif // TMPLANG_TREE_SOURCE_COMMONCONSTRUCTS_H
