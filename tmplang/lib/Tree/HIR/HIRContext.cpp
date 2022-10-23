#include <tmplang/Tree/HIR/HIRContext.h>

using namespace tmplang::hir;

HIRContext::HIRContext()
    : i32Type(BuiltinType::Kind::K_i32), UnitType(BuiltinType::Kind::K_Unit) {}
