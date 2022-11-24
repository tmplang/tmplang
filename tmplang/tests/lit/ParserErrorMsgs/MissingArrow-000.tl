// RUN: not tmplangc %s 2>&1 | FileCheck %s 

fn foo i32 {}

// CHECK:error: missing type arrow
// CHECK:fn foo -> i32 {}
