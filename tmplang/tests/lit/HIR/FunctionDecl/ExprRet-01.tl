// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo {
  ret;
}

//CHECK:ExprRet: ()
