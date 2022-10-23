// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

fn foo {}

//      CHECK:CompilationUnit ({{.+}})
// CHECK-NEXT:FunctionDecl ({{.+}}) fn foo -> ()
