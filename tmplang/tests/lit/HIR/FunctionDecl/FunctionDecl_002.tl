// RUN: tmplangc %s --dump-hir 2>&1 | FileCheck %s

fn foo: i32 a -> i32 {}

//      CHECK:CompilationUnit ({{.+}})
// CHECK-NEXT:FunctionDecl ({{.+}}) fn foo -> i32
// CHECK-NEXT:ParameterDecl ({{.+}}) i32 a
