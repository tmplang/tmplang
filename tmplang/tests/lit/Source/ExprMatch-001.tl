// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

proc foo {
  match a {
    (_, 25) -> 1,
    (a, 1) -> 2,
    ((a), b) -> 3,
    (10, (10, b)) -> 4,
    (_, _, _, a, 10) -> 5,
    otherwise -> 6
  };
}

//      CHECK: ExprMatch: match { }
// CHECK-NEXT: |-ExprVarRef: a
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-TupleDestructuration: ()
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-VoidPlaceholder:
// CHECK-NEXT: | | `-TupleDestructurationElem:
// CHECK-NEXT: | |   `-ExprIntegerNumber: 25
// CHECK-NEXT: | `-ExprIntegerNumber: 1
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-TupleDestructuration: ()
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-PlaceholderDecl: a
// CHECK-NEXT: | | `-TupleDestructurationElem:
// CHECK-NEXT: | |   `-ExprIntegerNumber: 1
// CHECK-NEXT: | `-ExprIntegerNumber: 2
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-TupleDestructuration: ()
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-TupleDestructuration: ()
// CHECK-NEXT: | | |   `-TupleDestructurationElem:
// CHECK-NEXT: | | |     `-PlaceholderDecl: a
// CHECK-NEXT: | | `-TupleDestructurationElem:
// CHECK-NEXT: | |   `-PlaceholderDecl: b
// CHECK-NEXT: | `-ExprIntegerNumber: 3
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-TupleDestructuration: ()
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-ExprIntegerNumber: 10
// CHECK-NEXT: | | `-TupleDestructurationElem:
// CHECK-NEXT: | |   `-TupleDestructuration: ()
// CHECK-NEXT: | |     |-TupleDestructurationElem: ,
// CHECK-NEXT: | |     | `-ExprIntegerNumber: 10
// CHECK-NEXT: | |     `-TupleDestructurationElem:
// CHECK-NEXT: | |       `-PlaceholderDecl: b
// CHECK-NEXT: | `-ExprIntegerNumber: 4
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-TupleDestructuration: ()
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-VoidPlaceholder:
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-VoidPlaceholder:
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-VoidPlaceholder:
// CHECK-NEXT: | | |-TupleDestructurationElem: ,
// CHECK-NEXT: | | | `-PlaceholderDecl: a
// CHECK-NEXT: | | `-TupleDestructurationElem:
// CHECK-NEXT: | |   `-ExprIntegerNumber: 10
// CHECK-NEXT: | `-ExprIntegerNumber: 5
// CHECK-NEXT: `-ExprMatchCase: ->
// CHECK-NEXT:   |-Otherwise:
// CHECK-NEXT:   `-ExprIntegerNumber: 6
