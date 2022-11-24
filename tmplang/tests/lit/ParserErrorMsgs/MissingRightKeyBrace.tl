// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo {

// CHECK:error: missing right key brace
// CHECK:proc foo { }
