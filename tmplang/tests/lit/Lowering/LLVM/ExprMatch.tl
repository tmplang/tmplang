// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

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

// CHECK:%Dummy = type { i32, i32 }

//       CHECK:define i32 @foo({ i32, %Dummy } %0) {
//  CHECK-NEXT:  br label %2
// CHECK-EMPTY:
//  CHECK-NEXT:2:                                                ; preds = %1
//  CHECK-NEXT:  %3 = extractvalue { i32, %Dummy } %0, 0
//  CHECK-NEXT:  %4 = icmp eq i32 %3, 1
//  CHECK-NEXT:  br i1 %4, label %6, label %5
// CHECK-EMPTY:
//  CHECK-NEXT:5:                                                ; preds = %2
//  CHECK-NEXT:  br label %25
// CHECK-EMPTY:
//  CHECK-NEXT:6:                                                ; preds = %2
//  CHECK-NEXT:  %7 = extractvalue { i32, %Dummy } %0, 1
//  CHECK-NEXT:  %8 = extractvalue %Dummy %7, 0
//  CHECK-NEXT:  %9 = icmp eq i32 %8, 3
//  CHECK-NEXT:  br i1 %9, label %11, label %10
// CHECK-EMPTY:
//  CHECK-NEXT:10:                                               ; preds = %6
//  CHECK-NEXT:  br label %25
// CHECK-EMPTY:
//  CHECK-NEXT:11:                                               ; preds = %6
//  CHECK-NEXT:  %12 = extractvalue { i32, %Dummy } %0, 0
//  CHECK-NEXT:  %13 = icmp eq i32 %12, 5
//  CHECK-NEXT:  br i1 %13, label %20, label %14
// CHECK-EMPTY:
//  CHECK-NEXT:14:                                               ; preds = %11
//  CHECK-NEXT:  %15 = extractvalue { i32, %Dummy } %0, 1
//  CHECK-NEXT:  %16 = extractvalue %Dummy %15, 1
//  CHECK-NEXT:  %17 = extractvalue %Dummy %15, 0
//  CHECK-NEXT:  %18 = icmp eq i32 %17, 10
//  CHECK-NEXT:  br i1 %18, label %20, label %19
// CHECK-EMPTY:
//  CHECK-NEXT:19:                                               ; preds = %14
//  CHECK-NEXT:  br label %25
// CHECK-EMPTY:
//  CHECK-NEXT:20:                                               ; preds = %14, %11
//  CHECK-NEXT:  %21 = extractvalue { i32, %Dummy } %0, 0
//  CHECK-NEXT:  %22 = icmp eq i32 %21, 6
//  CHECK-NEXT:  br i1 %22, label %24, label %23
// CHECK-EMPTY:
//  CHECK-NEXT:23:                                               ; preds = %20
//  CHECK-NEXT:  br label %25
// CHECK-EMPTY:
//  CHECK-NEXT:24:                                               ; preds = %20
//  CHECK-NEXT:  br label %25
// CHECK-EMPTY:
//  CHECK-NEXT:25:                                               ; preds = %24, %23, %19, %10, %5
//  CHECK-NEXT:  %26 = phi i32 [ 2, %5 ], [ 4, %10 ], [ %16, %19 ], [ 7, %23 ], [ 8, %24 ]
//  CHECK-NEXT:  ret i32 %26
//  CHECK-NEXT:}
