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

/// Severity level of the diagnostic
enum class HintKind : std::uint8_t {
  PreprendHint = 0,
};

/// Simple diagnostic builder. It is intended to be used to be printed as soon
/// it is created
class Hint {
public:
  Hint(HintKind kind) : Kind(kind) {}
  virtual ~Hint() = default;

  HintKind getKind() const { return Kind; }

  virtual void print(llvm::raw_ostream &, const SourceManager &) const = 0;

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

} // namespace tmplang

#endif // TMPLANG_DIAGNOSTIC_HINT_H
