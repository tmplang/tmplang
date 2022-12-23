// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo {
  (5 ());
}

// CHECK:error: missing comma before next tuple parameter
// CHECK:(5 , ());
