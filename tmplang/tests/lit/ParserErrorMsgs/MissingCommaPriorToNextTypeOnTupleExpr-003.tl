// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

proc foo {
  (() ());
}

// CHECK:error: missing comma before next tuple parameter
// CHECK:(() , ());
