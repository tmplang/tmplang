// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo: (i32  ()) var {}

// CHECK:error: missing comma before next tuple parameter
// CHECK:proc foo: (i32 , ()) var {}
