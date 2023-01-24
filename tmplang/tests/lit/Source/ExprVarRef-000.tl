// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

fn foo {
  hello;
}

// CHECK:ExprVarRef: hello
