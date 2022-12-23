// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo {
  (4 4);
}

// CHECK:error: missing comma before next tuple parameter
// CHECK:(4 , 4);
