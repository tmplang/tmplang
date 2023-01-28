// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

fn foo: A a {
  a.a;
}

//      CHECK:ExprDataFieldAccess: .a
// CHECK-NEXT:ExprVarRef: a
