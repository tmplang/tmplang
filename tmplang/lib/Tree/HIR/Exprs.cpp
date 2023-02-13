#include <tmplang/Tree/HIR/Exprs.h>

using namespace tmplang::hir;

const ExprMatchCaseLhsVal &AggregateDestructurationElem::getValue() const {
  return *Value;
}
