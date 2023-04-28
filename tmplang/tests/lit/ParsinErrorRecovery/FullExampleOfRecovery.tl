fn : i32 , i32 var i32 var2 -> i32

// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

//      CHECK:CompilationUnit:
// CHECK-NEXT:SubprogramDecl: fn Err![<subprogram_identifier>] : -> i32 Err![{] Err![}]
// CHECK-NEXT:ParamDecl: i32 Err![<parameter_id>] ,
// CHECK-NEXT:ParamDecl: i32 var Err![,]
// CHECK-NEXT:ParamDecl: i32 var2

// XFAIL: *
