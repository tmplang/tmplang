// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

data A = 
  a : (i32, ()),
  b : i32;
 
data B = 
  a : (A, A);

fn foo: B a {
  ret a.a.1.a.1;
}

//      CHECK:ExprAggregateDataAccess: 1 ()
// CHECK-NEXT:ExprAggregateDataAccess: a (i32, ())
// CHECK-NEXT:ExprAggregateDataAccess: 1 A{...}
// CHECK-NEXT:ExprAggregateDataAccess: a (A{...}, A{...})
// CHECK-NEXT:ExprVarRef: a B{...}
