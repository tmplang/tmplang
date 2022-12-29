// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

fn foo -> i32 {
  ret 1;
}

//      CHECK:define i32 @foo()
// CHECK-NEXT:  ret i32 1
// CHECK-NEXT:}
