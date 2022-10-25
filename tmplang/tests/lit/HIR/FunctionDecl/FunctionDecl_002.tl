// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo: i32 a -> i32 {}

//      CHECK:CompilationUnit:
// CHECK-NEXT:FuncDecl: fn foo -> i32
// CHECK-NEXT:ParamDecl: i32 a
