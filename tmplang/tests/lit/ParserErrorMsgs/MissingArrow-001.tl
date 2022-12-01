// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo: i32 a () {}

// CHECK:error: missing type arrow
// CHECK:proc foo: i32 a -> () {}
