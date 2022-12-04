// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

proc foo: (() ()) var {}

// CHECK:error: missing comma before next tuple parameter
// CHECK:proc foo: (() , ()) var {}
