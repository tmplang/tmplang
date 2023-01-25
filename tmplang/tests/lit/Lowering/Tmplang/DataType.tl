// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

data Dummy =
  a : i32;

fn foo: Dummy a {}

// CHECK:tmplang.subprogram private @foo(%arg0: !tmplang<data "Dummy"{i32}>) -> tuple<> {
