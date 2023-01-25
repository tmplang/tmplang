// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo -> i32 {
  ret 1;
}

//      CHECK:tmplang.subprogram private @foo() -> i32 {
// CHECK-NEXT:  %0 = tmplang.constant 1 -> i32
// CHECK-NEXT:  tmplang.return(%0) -> i32
// CHECK-NEXT:^bb1:  // no predecessors
// CHECK-NEXT:}
