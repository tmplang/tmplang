// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

proc foo: i32 a () {}

// CHECK:FuncDecl: proc foo : Err![->] () { }
