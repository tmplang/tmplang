// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo: i32 a, i32 b -> i32 {}

//      CHECK:CompilationUnit:
// CHECK-NEXT:SubprogramDecl: fn foo <i32, i32> -> i32
// CHECK-NEXT:ParamDecl: i32 a
// CHECK-NEXT:ParamDecl: i32 b
