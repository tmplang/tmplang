// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

proc foo {}

//      CHECK:CompilationUnit {{.*}}:
// CHECK-NEXT:FuncDecl {{.*}} <3,1>-<3,11>: proc foo -> ()
