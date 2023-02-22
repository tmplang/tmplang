// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

data Dummy =
 bar : i32,
 foo : i32;

proc foo: (i32, Dummy) a -> i32 {
  ret match a {
    (1, a) -> 2,
    (a, {bar: 3}) -> 4,
    (5, {foo : b, bar : 10}) -> b,
    (6, b) -> 7,
    _ -> 8,
    otherwise -> 9
  };
}

//      CHECK:%0 = tmplang.match(%arg0) : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> i32 {
// CHECK-NEXT:  %1 = tmplang.aggregateDataAccess(%arg0), 0 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> i32
// CHECK-NEXT:  %2 = tmplang.constant 1 -> i32
// CHECK-NEXT:  %3 = arith.cmpi eq, %2, %1 : i32
// CHECK-NEXT:  cf.cond_br %3, ^bb2, ^bb1
// CHECK-NEXT:^bb1:  // pred: ^bb0
// CHECK-NEXT:  %4 = tmplang.aggregateDataAccess(%arg0), 1 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> !tmplang<data "Dummy"{i32, i32}>
// CHECK-NEXT:  %5 = tmplang.constant 2 -> i32
// CHECK-NEXT:  tmplang.match_yield(%5) : i32
// CHECK-NEXT:^bb2:  // pred: ^bb0
// CHECK-NEXT:  %6 = tmplang.aggregateDataAccess(%arg0), 0 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> i32
// CHECK-NEXT:  %7 = tmplang.aggregateDataAccess(%arg0), 1 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> !tmplang<data "Dummy"{i32, i32}>
// CHECK-NEXT:  %8 = tmplang.aggregateDataAccess(%7), 0 : !tmplang<data "Dummy"{i32, i32}> -> i32
// CHECK-NEXT:  %9 = tmplang.constant 3 -> i32
// CHECK-NEXT:  %10 = arith.cmpi eq, %9, %8 : i32
// CHECK-NEXT:  cf.cond_br %10, ^bb4, ^bb3
// CHECK-NEXT:^bb3:  // pred: ^bb2
// CHECK-NEXT:  %11 = tmplang.constant 4 -> i32
// CHECK-NEXT:  tmplang.match_yield(%11) : i32
// CHECK-NEXT:^bb4:  // pred: ^bb2
// CHECK-NEXT:  %12 = tmplang.aggregateDataAccess(%arg0), 0 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> i32
// CHECK-NEXT:  %13 = tmplang.constant 5 -> i32
// CHECK-NEXT:  %14 = arith.cmpi eq, %13, %12 : i32
// CHECK-NEXT:  cf.cond_br %14, ^bb7, ^bb5
// CHECK-NEXT:^bb5:  // pred: ^bb4
// CHECK-NEXT:  %15 = tmplang.aggregateDataAccess(%arg0), 1 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> !tmplang<data "Dummy"{i32, i32}>
// CHECK-NEXT:  %16 = tmplang.aggregateDataAccess(%15), 1 : !tmplang<data "Dummy"{i32, i32}> -> i32
// CHECK-NEXT:  %17 = tmplang.aggregateDataAccess(%15), 0 : !tmplang<data "Dummy"{i32, i32}> -> i32
// CHECK-NEXT:  %18 = tmplang.constant 10 -> i32
// CHECK-NEXT:  %19 = arith.cmpi eq, %18, %17 : i32
// CHECK-NEXT:  cf.cond_br %19, ^bb7, ^bb6
// CHECK-NEXT:^bb6:  // pred: ^bb5
// CHECK-NEXT:  tmplang.match_yield(%16) : i32
// CHECK-NEXT:^bb7:  // 2 preds: ^bb4, ^bb5
// CHECK-NEXT:  %20 = tmplang.aggregateDataAccess(%arg0), 0 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> i32
// CHECK-NEXT:  %21 = tmplang.constant 6 -> i32
// CHECK-NEXT:  %22 = arith.cmpi eq, %21, %20 : i32
// CHECK-NEXT:  cf.cond_br %22, ^bb9, ^bb8
// CHECK-NEXT:^bb8:  // pred: ^bb7
// CHECK-NEXT:  %23 = tmplang.aggregateDataAccess(%arg0), 1 : tuple<i32, !tmplang<data "Dummy"{i32, i32}>> -> !tmplang<data "Dummy"{i32, i32}>
// CHECK-NEXT:  %24 = tmplang.constant 7 -> i32
// CHECK-NEXT:  tmplang.match_yield(%24) : i32
// CHECK-NEXT:^bb9:  // pred: ^bb7
// CHECK-NEXT:  %25 = tmplang.constant 8 -> i32
// CHECK-NEXT:  tmplang.match_yield(%25) : i32
// CHECK-NEXT:^bb10:  // no predecessors
// CHECK-NEXT:  %26 = tmplang.constant 9 -> i32
// CHECK-NEXT:  tmplang.match_yield(%26) : i32
// CHECK-NEXT:}
