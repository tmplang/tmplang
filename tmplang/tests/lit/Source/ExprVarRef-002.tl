// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

fn foo {
  ret hello;
}

//      CHECK:`-ExprRet:
// CHECK-NEXT:  `-ExprVarRef: hello
