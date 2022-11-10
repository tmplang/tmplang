// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo -> (i32) {}
fn bar: (i32, i32) pair -> (i32, i32) {}
fn baz -> (i32, (i32, i32 ,i32), (i32), ()) {}
fn zaz: ((((((((((())))))))))) muchParentheses -> (i32, (i32, i32 ,i32), (i32), ()) {}

//     CHECK:CompilationUnit:
//CHECK-NEXT:|-FuncDecl: fn foo -> (i32)
//CHECK-NEXT:|-FuncDecl: fn bar -> (i32, i32)
//CHECK-NEXT:| `-ParamDecl: (i32, i32) pair
//CHECK-NEXT:|-FuncDecl: fn baz -> (i32, (i32, i32, i32), (i32), ())
//CHECK-NEXT:`-FuncDecl: fn zaz -> (i32, (i32, i32, i32), (i32), ())
//CHECK-NEXT:  `-ParamDecl: ((((((((((())))))))))) muchParentheses
