#ifndef TMPLANG_DIAGNOSTIC_HINT_H
#define TMPLANG_DIAGNOSTIC_HINT_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <tmplang/Lexer/SourceLocation.h>

namespace llvm {
class raw_ostream;
} // namespace llvm

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

  virtual void print(llvm::raw_ostream &, const SourceManager &) const {};

private:
  HintKind Kind;
};

class PreprendHint : public Hint {
public:
  PreprendHint(SourceLocation srcLoc, llvm::ArrayRef<llvm::StringRef> hints)
      : Hint(HintKind::PreprendHint), SrcLoc(srcLoc), Hints(std::move(hints)) {
    assert(!Hints.empty());
  }

  void print(llvm::raw_ostream &, const SourceManager &) const override;

  static bool classof(const Hint *h) {
    return h->getKind() == HintKind::PreprendHint;
  }

private:
  SourceLocation SrcLoc;
  llvm::ArrayRef<llvm::StringRef> Hints;
};

class InsertTextAtHint : public Hint {
public:
  InsertTextAtHint(SourceLocation srcLoc, llvm::StringRef txt,
                   llvm::StringRef requiredLSep = " ",
                   llvm::StringRef requiredRSep = " ")
      : Hint(HintKind::PreprendHint), SrcLoc(srcLoc), TextToInsert(txt),
        RequiredLSep(requiredLSep), RequiredRSep(requiredRSep) {
    assert(!txt.empty());
  }

  void print(llvm::raw_ostream &, const SourceManager &) const override;

  static bool classof(const Hint *h) {
    return h->getKind() == HintKind::PreprendHint;
  }

private:
  SourceLocation SrcLoc;
  llvm::StringRef TextToInsert;
  llvm::StringRef RequiredLSep;
  llvm::StringRef RequiredRSep;
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
