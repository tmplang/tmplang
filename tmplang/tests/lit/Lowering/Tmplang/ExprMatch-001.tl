// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

union MyUnion =
  one (val: i32),
  two (value: i32);

union MyUnion2 =
  firstAlternative  (value: MyUnion),
  secondAlternative (firstVal : i32, secondVal : (i32, i32));

fn foo: MyUnion2 a {
  match a {
    firstAlternative{value: two{value: 1} } -> 2,
    secondAlternative{firstVal: 3, secondVal: snd} -> 4,
    a -> 5,
    otherwise -> 6
  };
}

//      CHECK:  %2 = tmplang.unionAlternativeCheck(%arg0), 0 : !tmplang<Union "MyUnion2"{ "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>},  "secondAlternative"{i32, tuple<i32, i32>}}> -> i1
// CHECK-NEXT:  cf.cond_br %2, ^bb4, ^bb1
// CHECK-NEXT:^bb1:  // pred: ^bb0
// CHECK-NEXT:  %3 = "tmplang.unionAccess"(%arg0) {alternativeType = i1} : (!tmplang<Union "MyUnion2"{ "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>},  "secondAlternative"{i32, tuple<i32, i32>}}>) -> !tmplang<data "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>}>
// CHECK-NEXT:  %4 = tmplang.aggregateDataAccess(%3), 0 : !tmplang<data "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>}> -> !tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>
// CHECK-NEXT:  %5 = tmplang.unionAlternativeCheck(%4), 1 : !tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}> -> i1
// CHECK-NEXT:  cf.cond_br %5, ^bb4, ^bb2
// CHECK-NEXT:^bb2:  // pred: ^bb1

//      CHECK:  %11 = tmplang.unionAlternativeCheck(%arg0), 1 : !tmplang<Union "MyUnion2"{ "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>},  "secondAlternative"{i32, tuple<i32, i32>}}> -> i1
// CHECK-NEXT:  cf.cond_br %11, ^bb7, ^bb5
// CHECK-NEXT:^bb5:  // pred: ^bb4
// CHECK-NEXT:  %12 = "tmplang.unionAccess"(%arg0) {alternativeType = i1} : (!tmplang<Union "MyUnion2"{ "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>},  "secondAlternative"{i32, tuple<i32, i32>}}>) -> !tmplang<data "secondAlternative"{i32, tuple<i32, i32>}>
