// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

proc foo {}

//      CHECK:CompilationUnit:
// CHECK-NEXT:FuncDecl: proc foo -> ()
