#include <tmplang/Lowering/Dialect/HIR/EnumAttrs.h>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

using namespace tmplang;

void TmplangHIRDialect::registerAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include <tmplang/Lowering/Dialect/HIR/TmplangHIROpsAttributes.cpp.inc>
      >();
}

// Include enum and attribute definitions
#include <tmplang/Lowering/Dialect/HIR/TmplangHIROpsEnum.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <tmplang/Lowering/Dialect/HIR/TmplangHIROpsAttributes.cpp.inc>
