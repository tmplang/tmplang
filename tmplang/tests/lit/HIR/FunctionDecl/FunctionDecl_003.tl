// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

fn foo: i32 a, i32 b -> i32 {}

//      CHECK:CompilationUnit {{.*}}:
// CHECK-NEXT:FuncDecl {{.*}} <3,1>-<3,30>: fn foo -> i32
// CHECK-NEXT:ParamDecl {{.*}} <3,9>-<3,13>: i32 a
// CHECK-NEXT:ParamDecl {{.*}} <3,16>-<3,20>: i32 b
