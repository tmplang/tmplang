// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

data A = 
  one : i32,
  two : i32;

fn foo: (A, i32) a {
  a.0.one;
//      CHECK:%0 = tmplang.aggregateDataAccess(%arg0), 0 : tuple<!tmplang<data "A"{i32, i32}>, i32> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT:%1 = tmplang.aggregateDataAccess(%0), 0 : !tmplang<data "A"{i32, i32}> -> i32

  a.0.two;
// CHECK-NEXT:%2 = tmplang.aggregateDataAccess(%arg0), 0 : tuple<!tmplang<data "A"{i32, i32}>, i32> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT:%3 = tmplang.aggregateDataAccess(%2), 1 : !tmplang<data "A"{i32, i32}> -> i32

  a.1;
// CHECK-NEXT:%4 = tmplang.aggregateDataAccess(%arg0), 1 : tuple<!tmplang<data "A"{i32, i32}>, i32> -> i32
}
