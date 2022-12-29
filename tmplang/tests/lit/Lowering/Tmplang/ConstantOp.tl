// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo {
  1;
}

// CHECK:%0 = tmplang.constant 1 -> i32
