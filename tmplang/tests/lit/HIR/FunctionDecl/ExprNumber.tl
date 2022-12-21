// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo {
  47;
}

//CHECK:ExprIntegerNumber: i32 : 47
