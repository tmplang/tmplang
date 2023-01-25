// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo -> (i32, i32) {
  ret (1, 2);
}

//      CHECK:tmplang.subprogram private @foo() -> tuple<i32, i32> {
// CHECK-NEXT:  %0 = tmplang.constant 1 -> i32
// CHECK-NEXT:  %1 = tmplang.constant 2 -> i32
// CHECK-NEXT:  %2 = tmplang.tuple(%0, %1) : (i32, i32) -> tuple<i32, i32>
// CHECK-NEXT:  tmplang.return(%2) -> tuple<i32, i32>
// CHECK-NEXT:^bb1:  // no predecessors
// CHECK-NEXT:}
