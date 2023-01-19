// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

fn foo -> (i32) {}
fn bar: (i32, i32) pair -> (i32, i32) {}
fn baz -> (i32, (i32, i32 ,i32), (i32), ()) {}
fn zaz: ((((((((((())))))))))) muchParentheses -> (i32, (i32, i32 ,i32), (i32), ()) {}

//     CHECK:CompilationUnit:
//CHECK-NEXT:|-SubprogramDecl: fn foo <> -> (i32)
//CHECK-NEXT:|-SubprogramDecl: fn bar <(i32, i32)> -> (i32, i32)
//CHECK-NEXT:| `-ParamDecl: (i32, i32) pair
//CHECK-NEXT:|-SubprogramDecl: fn baz <> -> (i32, (i32, i32, i32), (i32), ())
//CHECK-NEXT:`-SubprogramDecl: fn zaz <((((((((((()))))))))))> -> (i32, (i32, i32, i32), (i32), ())
//CHECK-NEXT:  `-ParamDecl: ((((((((((())))))))))) muchParentheses
