// RUN: not tmplangc %s 2>&1 | FileCheck %s 

fn foo -> {}

// CHECK:error: missing type
// CHECK:fn foo -> <type> {}
