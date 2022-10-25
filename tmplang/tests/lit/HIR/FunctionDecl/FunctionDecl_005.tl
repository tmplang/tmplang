// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo {}

//      CHECK:CompilationUnit:
// CHECK-NEXT:FuncDecl: fn foo -> ()
