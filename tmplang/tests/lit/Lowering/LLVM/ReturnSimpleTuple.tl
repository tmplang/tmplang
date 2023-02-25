// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

fn foo -> (i32) {
  ret (1);
}

//      CHECK:define { i32 } @foo()
// CHECK-NEXT:  %1 = alloca { i32 }, i64 1, align 8
// CHECK-NEXT:  %2 = getelementptr { i32 }, ptr %1, i64 0
// CHECK-NEXT:  store i32 1, ptr %2, align 4
// CHECK-NEXT:  %3 = load { i32 }, ptr %1, align 4
// CHECK-NEXT:  ret { i32 } %3
// CHECK-NEXT:}
