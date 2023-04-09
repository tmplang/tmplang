#ifndef TMPLANG_LOWERING_SEMA_SEMA_H
#define TMPLANG_LOWERING_SEMA_SEMA_H

#include <tmplang/Lowering/Dialect/HIR/Ops.h>

namespace tmplang {

class SourceManager;

/// Resolves symbols (identifier), types, and reports inconsistencies such as:
///   - Resolving of types
///   - Validation of duplication of symbols 
///   - Etc
/// The parameter SourceManager allow us to report back to the user pointing to
/// the source code
bool Sema(TranslationUnitOp mod, const SourceManager &sm);

} // namespace tmplang

#endif // TMPLANG_LOWERING_SEMA_SEMA_H
