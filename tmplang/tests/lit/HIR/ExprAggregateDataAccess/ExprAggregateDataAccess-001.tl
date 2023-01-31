// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

data A = 
  a : i32,
  b : i32;
 
data B = 
  a : A;

fn foo: B a {
  a.a.a;
  a.a.b;
}

//      CHECK:ExprAggregateDataAccess: a i32
// CHECK-NEXT:ExprAggregateDataAccess: a A{...}
// CHECK-NEXT:ExprVarRef: a B{...}
// CHECK-NEXT:ExprAggregateDataAccess: b i32
// CHECK-NEXT:ExprAggregateDataAccess: a A{...}
// CHECK-NEXT:ExprVarRef: a B{...}
