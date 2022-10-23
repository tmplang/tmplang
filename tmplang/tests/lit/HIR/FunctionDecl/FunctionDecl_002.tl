// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

fn foo: i32 a -> i32 {}

//      CHECK:CompilationUnit {{.*}}:
// CHECK-NEXT:FuncDecl {{.*}} <3,1>-<3,23>: fn foo -> i32
// CHECK-NEXT:ParamDecl {{.*}} <3,9>-<3,13>: i32 a
