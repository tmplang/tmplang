// RUN: not tmplangc %s 2>&1 | FileCheck %s 

fn foo: (()  i32) var {}

// CHECK:error: missing comma before next tuple parameter
// CHECK:fn foo: (() , i32) var {}
