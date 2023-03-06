// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

proc foo {
  match a {
    firstAlternative{ bar: a, foo: b } -> 1,
    secondAlternative{ bar: (_ , _), foo: signedAlternative{ num: 5 } } -> 2,
    otherwise -> 10
  };
}

//      CHECK:ExprMatch: match { }
// CHECK-NEXT:|-ExprVarRef: a
// CHECK-NEXT:|-ExprMatchCase: ->
// CHECK-NEXT:| |-UnionDestructuration: firstAlternative
// CHECK-NEXT:| | `-DataDestructuration: {}
// CHECK-NEXT:| |   |-DataDestructurationElem: bar : ,
// CHECK-NEXT:| |   | `-PlaceholderDecl: a
// CHECK-NEXT:| |   `-DataDestructurationElem: foo :
// CHECK-NEXT:| |     `-PlaceholderDecl: b
// CHECK-NEXT:| `-ExprIntegerNumber: 1
// CHECK-NEXT:|-ExprMatchCase: ->
// CHECK-NEXT:| |-UnionDestructuration: secondAlternative
// CHECK-NEXT:| | `-DataDestructuration: {}
// CHECK-NEXT:| |   |-DataDestructurationElem: bar : ,
// CHECK-NEXT:| |   | `-TupleDestructuration: ()
// CHECK-NEXT:| |   |   |-TupleDestructurationElem: ,
// CHECK-NEXT:| |   |   | `-VoidPlaceholder:
// CHECK-NEXT:| |   |   `-TupleDestructurationElem:
// CHECK-NEXT:| |   |     `-VoidPlaceholder:
// CHECK-NEXT:| |   `-DataDestructurationElem: foo :
// CHECK-NEXT:| |     `-UnionDestructuration: signedAlternative
// CHECK-NEXT:| |       `-DataDestructuration: {}
// CHECK-NEXT:| |         `-DataDestructurationElem: num :
// CHECK-NEXT:| |           `-ExprIntegerNumber: 5
// CHECK-NEXT:| `-ExprIntegerNumber: 2
// CHECK-NEXT:`-ExprMatchCase: ->
// CHECK-NEXT:  |-Otherwise:
// CHECK-NEXT:  `-ExprIntegerNumber: 10
