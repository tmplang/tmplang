// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

fn foo { 5 5; }

// CHECK:ExprStmt: Err![;]
// CHECK:ExprNumber: 5
