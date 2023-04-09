#ifndef TMPLANG_LIB_LOWERING_SEMA_SYMBOLTYPERESOLVER_H
#define TMPLANG_LIB_LOWERING_SEMA_SYMBOLTYPERESOLVER_H

#include <tmplang/Lowering/Dialect/HIR/Ops.h>

namespace tmplang {

bool ResolveSymbolTypes(tmplang::TranslationUnitOp tuOp);

} // namespace tmplang

#endif // TMPLANG_LIB_LOWERING_SEMA_SYMBOLTYPERESOLVER_H
