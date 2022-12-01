// RUN: not tmplangc %s 2>&1 | FileCheck %s 


proc foo: ) var {}

// CHECK:error:  missing left parenthesis opening tuple
// CHECK:proc foo: ( ) var {}
