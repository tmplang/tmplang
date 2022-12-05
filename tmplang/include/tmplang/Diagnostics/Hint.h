#ifndef TMPLANG_DIAGNOSTIC_HINT_H
#define TMPLANG_DIAGNOSTIC_HINT_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/Lexer/SourceLocation.h>

namespace tmplang {

class SourceManager;

enum class HintKind : std::uint8_t {
  PreprendHint = 0,
  InsertTextAtHint,
  NoHint
};

/// Simple diagnostic builder. It is intended to be used to be printed as soon
/// it is created
class Hint {
public:
  Hint(HintKind kind) : Kind(kind) {}
  virtual ~Hint() = default;

  HintKind getKind() const { return Kind; }

  virtual void print(raw_ostream &, const SourceManager &) const {};

private:
  HintKind Kind;
};

class InsertTextAtHint : public Hint {
public:
  InsertTextAtHint(SourceLocation srcLoc, StringRef hint,
                   StringRef requiredLSep = " ", StringRef requiredRSep = " ",
                   StringRef subscriptMSg = "")
      : Hint(HintKind::InsertTextAtHint), SrcLoc(srcLoc), Hints(),
        RequiredLSep(requiredLSep), RequiredRSep(requiredRSep) {
    assert(!hint.empty());
    Hints.push_back(hint);
  }

  InsertTextAtHint(SourceLocation srcLoc, ArrayRef<StringRef> hints,
                   StringRef requiredLSep = " ", StringRef requiredRSep = " ")
      : Hint(HintKind::InsertTextAtHint), SrcLoc(srcLoc), Hints(hints),
        RequiredLSep(requiredLSep), RequiredRSep(requiredRSep) {
    assert(!Hints.empty());
    assert(llvm::none_of(Hints, [](StringRef hint) { return hint.empty(); }));
  }

  void print(raw_ostream &, const SourceManager &) const override;

  static bool classof(const Hint *h) {
    return h->getKind() == HintKind::InsertTextAtHint;
  }

private:
  SourceLocation SrcLoc;
  SmallVector<StringRef, 3> Hints;
  StringRef RequiredLSep;
  StringRef RequiredRSep;
};

class NoHint : public Hint {
public:
  NoHint() : Hint(HintKind::NoHint) {}

  static bool classof(const Hint *h) {
    return h->getKind() == HintKind::NoHint;
  }
};

} // namespace tmplang

#endif // TMPLANG_DIAGNOSTIC_HINT_H
