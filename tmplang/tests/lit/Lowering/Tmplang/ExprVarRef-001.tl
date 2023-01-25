// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

data A =
  a : i32,
  b : i32,
  c : i32;

fn foo: A a -> A {
  ret a;
}

//      CHECK:tmplang.subprogram private @foo(%arg0: !tmplang<data "A"{i32, i32, i32}>) -> !tmplang<data "A"{i32, i32, i32}> {
// CHECK-NEXT:  tmplang.return(%arg0) -> !tmplang<data "A"{i32, i32, i32}>
