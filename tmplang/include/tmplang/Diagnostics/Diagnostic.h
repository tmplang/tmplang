#ifndef TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H
#define TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Lexer/SourceLocation.h>

#include <bit>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace tmplang {

class SourceManager;

/// Simple wrapper for stderr that allows enable or not colors when printing
struct diagnostic_ostream final : public llvm::raw_fd_ostream {
  diagnostic_ostream(const bool colors)
      : raw_fd_ostream(STDERR_FILENO, /*shouldClose*/ false,
                       /*unbuffered*/ true) {
    enable_colors(colors);
  }
  ~diagnostic_ostream() = default;
};

/// Severity level of the diagnostic
enum class DiagnosticSeverity : std::uint8_t {
  Warning = 0,
  Error,
  MaxSeverityVal = Error
};
[[maybe_unused]] constexpr unsigned NumSeverityKinds =
    static_cast<unsigned>(DiagnosticSeverity::MaxSeverityVal) + 1;

llvm::StringLiteral ToString(DiagnosticSeverity);

enum class DiagId : std::uint32_t {
#define DIAG(ID, SEV, MSG) ID,
#include "DiagnosticMessages.def"
};

static inline constexpr struct {
  DiagnosticSeverity Sev;
  llvm::StringLiteral Msg;
} DiagnosticMessages[] = {
#define DIAG(ID, SEV, MSG) {SEV, MSG},
#include "DiagnosticMessages.def"
};

/// Simple diagnostic builder. It is intended to be used to be printed as soon
/// it is created
class Diagnostic {
public:
  Diagnostic(DiagId id, SourceLocationSpan locSpan)
      : Id(id), LocSpan(locSpan) {}

  Diagnostic() = delete;
  Diagnostic(const Diagnostic &) = delete;
  Diagnostic(Diagnostic &&) = delete;
  Diagnostic &operator=(const Diagnostic &) = delete;
  Diagnostic &operator=(Diagnostic &&) = delete;

  void print(llvm::raw_ostream &, const SourceManager &) const;

private:
  /*
    Format:
      <sev_str>:

    Example:
      error:
  */
  void printSeverity(llvm::raw_ostream &) const;

  /*
    Format:
      <msg>

    Example:
      missing function type on function definition
  */
  void printSummary(llvm::raw_ostream &) const;

  /*
    Format:
      <spaces_alignment_to_sev>at: <path>:<line>:<col>

    Example:
      error: blabla <- gets printed on ::printSeverity, it is just for context
         at: test.tmp:1:1 <- only this lines is printed by this function
      ^~~ <- <space_alignment_to_sev>
  */
  void printLocation(llvm::raw_ostream &, const SourceManager &sm) const;

  /*
  Format:
             |
   <lineNum> | <source_code>
             | <caret_and_tildes>
             |
  Example:
             |
           1 | fn bar: (i32, i32) a, (i32, i32) b -> (i32, i32) {
             | ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
           2 |
           3 | }
             | ~
             |
  */
  void printContext(llvm::raw_ostream &, const SourceManager &sm) const;

  DiagnosticSeverity getSeverity() const;
  llvm::StringRef getMessage() const;

private:
  // Prefer using the Id, so we don't have to store the message and the severity
  DiagId Id;
  SourceLocationSpan LocSpan;
};

} // namespace tmplang

#endif // TMPLANG_DIAGNOSTIC_DIAGNOSTIC_H
