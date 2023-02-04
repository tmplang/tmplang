// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

data A = 
  one : i32,
  two : i32;

fn foo: (A, i32) a -> i32 {
  ret a.0.one;
}

//      CHECK:define i32 @foo({ %A, i32 } %0)
// CHECK-NEXT:  %2 = extractvalue { %A, i32 } %0, 0
// CHECK-NEXT:  %3 = extractvalue %A %2, 0
// CHECK-NEXT:  ret i32 %3
// CHECK-NEXT:}
