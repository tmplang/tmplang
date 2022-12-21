// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo {
  ret 1;
}

//CHECK:ExprRet: i32
//CHECK:ExprIntegerNumber: i32 : 1
