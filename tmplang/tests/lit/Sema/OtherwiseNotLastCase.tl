// RUN: not tmplangc %s 2>&1 | FileCheck %s

proc foo: i32 a -> i32 {
  ret match a {
    1 -> 0, 
    otherwise -> 1,
    2 -> 1
  };
}

// CHECK: error: otherwise can only appear as the last case
