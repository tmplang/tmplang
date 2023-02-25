#ifndef TMPLANG_LOWERING_CONVESION_PASSES_H
#define TMPLANG_LOWERING_CONVESION_PASSES_H

#include <mlir/Pass/PassRegistry.h>

#include "TmplangToArith/TmplangToArith.h"
#include "TmplangToFunc/TmplangToFunc.h"
#include "TmplangToLLVM/TmplangToLLVM.h"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace tmplang {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <tmplang/Lowering/Conversion/Passes.h.inc>

} // namespace tmplang

#endif // TMPLANG_LOWERING_CONVESION_PASSES_H
