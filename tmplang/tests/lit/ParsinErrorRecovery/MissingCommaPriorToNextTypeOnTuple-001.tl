// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

proc foo: (i32 i32) var {}

// CHECK:ParamDecl: (i32 Err![,] i32) var
