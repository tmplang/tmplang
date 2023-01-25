// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo {}

//      CHECK:tmplang.subprogram private @foo() -> tuple<> {
// CHECK-NEXT:  %0 = tmplang.tuple() : () -> tuple<>
// CHECK-NEXT:  tmplang.return(%0) -> tuple<>
// CHECK-NEXT:}
