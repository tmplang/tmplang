#include <tmplang/Lowering/Sema/Sema.h>

#include "SymbolRedefinitionValidation.h"
#include "SymbolTypeResolver.h"
#include "SymbolUsersAndDerivedResolver.h"

using namespace tmplang;

bool tmplang::Sema(TranslationUnitOp tuOp, const SourceManager &) {
  return ValidateNoRedefinitionOfSymbols(tuOp) && ResolveSymbolTypes(tuOp) &&
         ResolveSymbolsUsersAndDerived(tuOp);
}