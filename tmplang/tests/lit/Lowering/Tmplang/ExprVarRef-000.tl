// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo: i32 a -> i32 {
  ret a;
}

//      CHECK:tmplang.subprogram private @foo(%arg0: i32) -> i32 {
// CHECK-NEXT:  tmplang.return(%arg0) -> i32
