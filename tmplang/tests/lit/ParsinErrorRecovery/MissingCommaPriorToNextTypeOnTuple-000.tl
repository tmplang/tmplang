// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

fn foo: (()  i32) var {}

// CHECK:ParamDecl: (() Err![,] i32) var
