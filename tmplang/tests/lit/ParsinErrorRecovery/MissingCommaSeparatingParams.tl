proc foo: i32 var i32 var2 i32 var3 {}

// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

//      CHECK:ParamDecl: i32 var Err![,]
// CHECK-NEXT:ParamDecl: i32 var2 Err![,]
// CHECK-NEXT:ParamDecl: i32 var3

// XFAIL:*
