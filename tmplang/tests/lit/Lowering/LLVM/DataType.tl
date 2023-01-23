// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

data Dummy =
  a : i32;

fn foo: Dummy a {}

// CHECK:%Dummy = type { i32 }
// CHECK:define {} @foo(%Dummy %0)
