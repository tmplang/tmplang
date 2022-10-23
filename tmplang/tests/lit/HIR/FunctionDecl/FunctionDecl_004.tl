// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

fn foo -> i32 {}

//      CHECK:CompilationUnit {{.*}}:
// CHECK-NEXT:FuncDecl {{.*}} <3,1>-<3,16>: fn foo -> i32
