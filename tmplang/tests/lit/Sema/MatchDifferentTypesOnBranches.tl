// RUN: not tmplangc %s 2>&1 | FileCheck %s

proc foo: i32 a -> i32 {
  ret match a {
    1 -> 0, 
    2 -> (), 
    otherwise -> (1)
  };
}

// CHECK: error: match expr branch type does not match first branch type
// CHECK: 6 |     2 -> (), 

// CHECK: error: match expr branch type does not match first branch type
// CHECK: 7 |     otherwise -> (1)
