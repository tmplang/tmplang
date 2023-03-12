// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo -> (i32) {
  ret (1);
}

//      CHECK:tmplang.subprogram private @foo() -> tuple<i32> {
// CHECK-NEXT:  %0 = tmplang.constant 1 -> i32
// CHECK-NEXT:  %1 = tmplang.tuple(%0) : (i32) -> tuple<i32>
// CHECK-NEXT:  tmplang.return(%1) -> tuple<i32>
// CHECK-NEXT:}
