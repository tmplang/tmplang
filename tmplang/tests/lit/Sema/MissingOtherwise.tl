// RUN: not tmplangc %s 2>&1 | FileCheck %s

proc foo: i32 a -> i32 {
  ret match a {
    1 -> 0, 
    2 -> 1 
  };
}

// CHECK: error: missing otherwise catch-all in match expr
