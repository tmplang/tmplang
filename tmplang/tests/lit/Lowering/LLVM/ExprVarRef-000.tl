// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

fn foo: i32 a -> i32 {
  ret a;
}

//      CHECK:define i32 @foo(i32 %0) !dbg !3 {
// CHECK-NEXT:  ret i32 %0, !dbg !7
