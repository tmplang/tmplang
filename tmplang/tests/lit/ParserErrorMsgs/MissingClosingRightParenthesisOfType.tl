// RUN: not tmplangc %s 2>&1 | FileCheck %s 


proc foo: (i32 var {}

// CHECK:error: missing right parenthesis closing tuple
// CHECK:proc foo: (i32 ) var {}
