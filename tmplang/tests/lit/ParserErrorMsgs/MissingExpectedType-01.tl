// RUN: not tmplangc %s 2>&1 | FileCheck %s 

fn foo: (i32, ) var {}

// CHECK:error: missing type
// CHECK:fn foo: (i32, <type> ) var {}
