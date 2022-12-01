// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo: i32 {}

// CHECK:error: missing variable identifier
// CHECK:proc foo: i32 <parameter_id> {}
