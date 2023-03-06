union MyUnion =
  one (val: i32),
  two (value: i32);

union MyUnion2 =
  firstAlternative  (value: MyUnion),
  secondAlternative (firstVal : i32, secondVal : (i32, i32));

fn foo: MyUnion2 a {
  match a {
    firstAlternative{value: one{val: 5} } -> 5,
    secondAlternative{firstVal: fst, secondVal: snd} -> 6,
    a -> 2,
    otherwise -> 1
  };
} 

// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

//      CHECK:ExprMatch: i32
// CHECK-NEXT:|-ExprVarRef: a MyUnion2{...}
// CHECK-NEXT:|-ExprMatchCase: i32
// CHECK-NEXT:| |-UnionDestructuration: firstAlternative{...} alternativeIdx: 0
// CHECK-NEXT:| | `-AggregateDestructuration: firstAlternative{...}
// CHECK-NEXT:| |   `-AggregateDestructurationElem: MyUnion{...}  idx: 0
// CHECK-NEXT:| |     `-UnionDestructuration: one{...} alternativeIdx: 0
// CHECK-NEXT:| |       `-AggregateDestructuration: one{...}
// CHECK-NEXT:| |         `-AggregateDestructurationElem: i32  idx: 0
// CHECK-NEXT:| |           `-ExprIntegerNumber: i32 : 5
// CHECK-NEXT:| `-ExprIntegerNumber: i32 : 5
// CHECK-NEXT:|-ExprMatchCase: i32
// CHECK-NEXT:| |-UnionDestructuration: secondAlternative{...} alternativeIdx: 1
// CHECK-NEXT:| | `-AggregateDestructuration: secondAlternative{...}
// CHECK-NEXT:| |   |-AggregateDestructurationElem: i32  idx: 0
// CHECK-NEXT:| |   | `-PlaceholderDecl: i32 : fst
// CHECK-NEXT:| |   `-AggregateDestructurationElem: (i32, i32)  idx: 1
// CHECK-NEXT:| |     `-PlaceholderDecl: (i32, i32) : snd
// CHECK-NEXT:| `-ExprIntegerNumber: i32 : 6
// CHECK-NEXT:|-ExprMatchCase: i32
// CHECK-NEXT:| |-PlaceholderDecl: MyUnion2{...} : a
// CHECK-NEXT:| `-ExprIntegerNumber: i32 : 2
// CHECK-NEXT:`-ExprMatchCase: i32
// CHECK-NEXT:  `-ExprIntegerNumber: i32 : 1
