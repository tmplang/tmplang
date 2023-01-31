// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

data A =
  a : i32;

fn foo: A a {
  a.a;
}

//      CHECK:ExprAggregateDataAccess: a i32
// CHECK-NEXT:ExprVarRef: a A{...}
