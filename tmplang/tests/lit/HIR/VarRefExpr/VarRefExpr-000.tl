// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo: i32 a {
  ret a;
}

// CHECK:ExprVarRef: a i32
