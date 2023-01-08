// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

fn foo i32 {}

// CHECK:SubprogramDecl: fn foo Err![->] i32 { }
