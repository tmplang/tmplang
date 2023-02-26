// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo: i32 var i32 var2 {}

// CHECK:error: missing needed comma between parameters
// CHECK:proc foo: i32 var , i32 var2 {}

// XFAIL: *
