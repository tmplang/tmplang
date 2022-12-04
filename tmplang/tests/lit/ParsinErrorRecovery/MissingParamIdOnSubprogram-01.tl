proc foo:() , i32 , i32 {}

// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

//      CHECK:ParamDecl: () Err![<parameter_id>]
// CHECK-NEXT:ParamDecl: i32 Err![<parameter_id>]
// CHECK-NEXT:ParamDecl: i32 Err![<parameter_id>]
