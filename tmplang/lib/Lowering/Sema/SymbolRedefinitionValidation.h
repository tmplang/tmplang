#ifndef TMPLANG_LIB_LOWERING_SEMA_SYMBOLREDEFINITIONVALIDATION_H
#define TMPLANG_LIB_LOWERING_SEMA_SYMBOLREDEFINITIONVALIDATION_H

#include <tmplang/Lowering/Dialect/HIR/Ops.h>

namespace tmplang {

/// Checks if there are redefinition of symbols
bool ValidateNoRedefinitionOfSymbols(tmplang::TranslationUnitOp tuOp);

} // namespace tmplang

#endif // TMPLANG_LIB_LOWERING_SEMA_SYMBOLREDEFINITIONVALIDATION_H
