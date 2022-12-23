// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s 

proc foo {
  (5;
}

// CHECK:error: missing right parenthesis closing tuple
// CHECK:(5 ) ;
