// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo {
  (1, 2, (3, 4), ());
}

//      CHECK:%0 = tmplang.constant 1 -> i32
// CHECK-NEXT:%1 = tmplang.constant 2 -> i32
// CHECK-NEXT:%2 = tmplang.constant 3 -> i32
// CHECK-NEXT:%3 = tmplang.constant 4 -> i32
// CHECK-NEXT:%4 = tmplang.tuple(%2, %3) : (i32, i32) -> tuple<i32, i32>
// CHECK-NEXT:%5 = tmplang.tuple() : () -> tuple<>
// CHECK-NEXT:%6 = tmplang.tuple(%0, %1, %4, %5) : (i32, i32, tuple<i32, i32>, tuple<>) -> tuple<i32, i32, tuple<i32, i32>, tuple<>>
