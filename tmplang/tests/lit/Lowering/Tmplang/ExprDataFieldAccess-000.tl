// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

data A = 
  a : i32,
  b : i32;

fn foo: A a -> i32 {
  ret a.a;
}

//      CHECK:%0 = tmplang.aggregateDataAccess(%arg0), 0 : !tmplang<data "A"{i32, i32}> -> i32
// CHECK-NEXT:  tmplang.return(%arg0) -> i32
