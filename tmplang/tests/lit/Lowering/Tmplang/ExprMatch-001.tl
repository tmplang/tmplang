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

//      CHECK:%2 = tmplang.unionAccess(%arg0), 0 : !tmplang<Union "MyUnion2"{ "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>},  "secondAlternative"{i32, tuple<i32, i32>}}> -> !tmplang<data "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>}>
// CHECK-NEXT:%3 = tmplang.aggregateDataAccess(%2), 0 : !tmplang<data "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>}> -> !tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>
// CHECK-NEXT:%4 = tmplang.unionAccess(%3), 1 : !tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}> -> !tmplang<data "two"{i32}>

// CHECK:%9 = tmplang.unionAccess(%arg0), 1 : !tmplang<Union "MyUnion2"{ "firstAlternative"{!tmplang<Union "MyUnion"{ "one"{i32},  "two"{i32}}>},  "secondAlternative"{i32, tuple<i32, i32>}}> -> !tmplang<data "secondAlternative"{i32, tuple<i32, i32>}>
