// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

proc foo {}

//      CHECK:CompilationUnit ({{.+}})
// CHECK-NEXT:FunctionDecl ({{.+}}) proc foo -> ()
