#ifndef TMPLANG_LIB_LOWERING_SEMA_SYMBOLUSERSANDDERIVEDRESOLVER_H
#define TMPLANG_LIB_LOWERING_SEMA_SYMBOLUSERSANDDERIVEDRESOLVER_H

#include <tmplang/Lowering/Dialect/HIR/Ops.h>

namespace tmplang {

/// Resolves all operation that uses symbols and derived ops such as
/// 'TupleAccessOp' or 'DataAccessOp'
bool ResolveSymbolsUsersAndDerived(tmplang::TranslationUnitOp tuOp);

} // namespace tmplang

#endif // TMPLANG_LIB_LOWERING_SEMA_SYMBOLUSERSANDDERIVEDRESOLVER_H
