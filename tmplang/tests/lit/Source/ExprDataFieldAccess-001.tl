// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

fn foo: A a {
  a.b.c.d.e;
}

//      CHECK:ExprDataFieldAccess: .e
// CHECK-NEXT:ExprDataFieldAccess: .d
// CHECK-NEXT:ExprDataFieldAccess: .c
// CHECK-NEXT:ExprDataFieldAccess: .b
// CHECK-NEXT:ExprVarRef: a
