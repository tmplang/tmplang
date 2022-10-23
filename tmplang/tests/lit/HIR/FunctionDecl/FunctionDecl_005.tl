// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

fn foo {}

//      CHECK:CompilationUnit {{.*}}:
// CHECK-NEXT:FuncDecl {{.*}} <3,1>-<3,9>: fn foo -> ()
