// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

data A =
  a : i32,
  b : i32,
  c : i32;

fn foo: A a -> A {
  ret a;
}

//      CHECK:%A = type { i32, i32, i32 }

//      CHECK:define %A @foo(%A %0)
// CHECK-NEXT:  ret %A %0
