// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

fn foo: (i32, (i32, i32, i32)) a {
  a.1.2;
}

//      CHECK:ExprAggregateDataAccess: .2
// CHECK-NEXT:ExprAggregateDataAccess: .1
// CHECK-NEXT:ExprVarRef: a
