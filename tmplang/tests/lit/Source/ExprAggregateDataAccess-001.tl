// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

fn foo: A a {
  a.b.c.d.e;
}

//      CHECK:ExprAggregateDataAccess: .e
// CHECK-NEXT:ExprAggregateDataAccess: .d
// CHECK-NEXT:ExprAggregateDataAccess: .c
// CHECK-NEXT:ExprAggregateDataAccess: .b
// CHECK-NEXT:ExprVarRef: a
