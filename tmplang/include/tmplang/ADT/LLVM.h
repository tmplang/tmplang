#ifndef TMPLANG_ADT_LLVM_H
#define TMPLANG_ADT_LLVM_H

// In case any more forwarding is need, take a look into mlir/Support/LLVM.h
// It could be already declared there, do some copy-pasterino

// We include these two headers because they cannot be practically forward
// declared, and are effectively language features.
#include <llvm/Support/Casting.h>

#if defined(__clang_major__)
#if __clang_major__ <= 5
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SmallVector.h"
#endif
#endif

// Forward declarations.
namespace llvm {
template <unsigned N> class SmallString;
class StringRef;
class StringLiteral;
class Twine;
template <typename T, typename R> class StringSwitch;

template <typename T> class ArrayRef;
template <typename T> class MutableArrayRef;
template <typename T, unsigned N> class SmallVector;
template <typename T> class SmallVectorImpl;

class raw_ostream;
class raw_string_ostream;
} // namespace llvm

#define AllForwardings                                                         \
  using llvm::cast;                                                            \
  using llvm::dyn_cast;                                                        \
  using llvm::isa;                                                             \
                                                                               \
  using llvm::SmallString;                                                     \
  using llvm::StringRef;                                                       \
  using llvm::StringLiteral;                                                   \
  using llvm::Twine;                                                           \
  using llvm::StringSwitch;                                                    \
                                                                               \
  using std::nullopt;                                                          \
                                                                               \
  using llvm::MutableArrayRef;                                                 \
  using llvm::ArrayRef;                                                        \
  using llvm::SmallVector;                                                     \
  using llvm::SmallVectorImpl;                                                 \
                                                                               \
  using llvm::raw_ostream;                                                     \
  using llvm::raw_string_ostream;

// Forward all llvm utilities to all tree namespaces
namespace tmplang {

AllForwardings

namespace hir {
  AllForwardings
} // namespace hir

namespace source {
AllForwardings
} // namespace source

} // namespace tmplang

#endif // TMPLANG_ADT_LLVM_H
