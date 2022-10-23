// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

fn foo -> i32 {}

//      CHECK:CompilationUnit ({{.+}})
// CHECK-NEXT:FunctionDecl ({{.+}}) fn foo -> i32
