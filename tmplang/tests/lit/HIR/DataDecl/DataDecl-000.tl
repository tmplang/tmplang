// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

data Dummy =
  a : i32,
  b : (i32, i32);

fn foo -> Dummy {}

//      CHECK:DataDecl:  Dummy
// CHECK-NEXT:|-DataFieldDecl: i32  a
// CHECK-NEXT:`-DataFieldDecl: (i32, i32)  b
// CHECK-NEXT:SubprogramDecl: fn foo <> -> Dummy{...}
