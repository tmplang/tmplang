// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

data Dummy =
  a : i32,
  b : (i32, i32);

data Dummy2 =
  a : Dummy;

fn foo: Dummy2 a -> Dummy {}

//      CHECK:DataDecl:  Dummy
// CHECK-NEXT:|-DataFieldDecl: i32  a
// CHECK-NEXT:`-DataFieldDecl: (i32, i32)  b
// CHECK-NEXT:DataDecl:  Dummy2
// CHECK-NEXT:`-DataFieldDecl: Dummy{...}  a
// CHECK-NEXT:SubprogramDecl: fn foo <Dummy2{...}> -> Dummy{...}
// CHECK-NEXT:`-ParamDecl: Dummy2{...} a
