// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

data A = 
  one : i32,
  two : i32;

data B = 
  three : A,
  four : A;

data C = 
  five : B,
  six : B;

fn foo: C a {
  a.five.three.one;
//      CHECK: %0 = tmplang.aggregateDataAccess(%arg0), 0 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %1 = tmplang.aggregateDataAccess(%0), 0 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %2 = tmplang.aggregateDataAccess(%1), 0 : !tmplang<data "A"{i32, i32}> -> i32

  a.five.three.two;
// CHECK-NEXT: %3 = tmplang.aggregateDataAccess(%arg0), 0 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %4 = tmplang.aggregateDataAccess(%3), 0 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %5 = tmplang.aggregateDataAccess(%4), 1 : !tmplang<data "A"{i32, i32}> -> i32

  a.five.four.one;
// CHECK-NEXT: %6 = tmplang.aggregateDataAccess(%arg0), 0 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %7 = tmplang.aggregateDataAccess(%6), 1 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %8 = tmplang.aggregateDataAccess(%7), 0 : !tmplang<data "A"{i32, i32}> -> i32

  a.five.four.two;
// CHECK-NEXT: %9 = tmplang.aggregateDataAccess(%arg0), 0 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %10 = tmplang.aggregateDataAccess(%9), 1 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %11 = tmplang.aggregateDataAccess(%10), 1 : !tmplang<data "A"{i32, i32}> -> i32

  a.six.three.one;
// CHECK-NEXT: %12 = tmplang.aggregateDataAccess(%arg0), 1 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %13 = tmplang.aggregateDataAccess(%12), 0 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %14 = tmplang.aggregateDataAccess(%13), 0 : !tmplang<data "A"{i32, i32}> -> i32

  a.six.three.two;
// CHECK-NEXT: %15 = tmplang.aggregateDataAccess(%arg0), 1 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %16 = tmplang.aggregateDataAccess(%15), 0 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %17 = tmplang.aggregateDataAccess(%16), 1 : !tmplang<data "A"{i32, i32}> -> i32
 
  a.six.four.one;
// CHECK-NEXT: %18 = tmplang.aggregateDataAccess(%arg0), 1 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %19 = tmplang.aggregateDataAccess(%18), 1 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %20 = tmplang.aggregateDataAccess(%19), 0 : !tmplang<data "A"{i32, i32}> -> i32

  a.six.four.two;
// CHECK-NEXT: %21 = tmplang.aggregateDataAccess(%arg0), 1 : !tmplang<data "C"{!tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>, !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>}> -> !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}>
// CHECK-NEXT: %22 = tmplang.aggregateDataAccess(%21), 1 : !tmplang<data "B"{!tmplang<data "A"{i32, i32}>, !tmplang<data "A"{i32, i32}>}> -> !tmplang<data "A"{i32, i32}>
// CHECK-NEXT: %23 = tmplang.aggregateDataAccess(%22), 1 : !tmplang<data "A"{i32, i32}> -> i32
}
