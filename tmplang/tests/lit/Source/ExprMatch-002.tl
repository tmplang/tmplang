// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

proc foo {
  match a {
    { bar: a, foo: b } -> 1,
    { bar: _, foo: _ } -> 2,
    { bar: { baz: 2 }, foo: _ } -> 3,
    otherwise -> 10
  };
}

//      CHECK: ExprMatch: match { }
// CHECK-NEXT: |-ExprVarRef: a
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-DataDestructuration: {}
// CHECK-NEXT: | | |-DataDestructurationElem: bar : ,
// CHECK-NEXT: | | | `-PlaceholderDecl: a
// CHECK-NEXT: | | `-DataDestructurationElem: foo :
// CHECK-NEXT: | |   `-PlaceholderDecl: b
// CHECK-NEXT: | `-ExprIntegerNumber: 1
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-DataDestructuration: {}
// CHECK-NEXT: | | |-DataDestructurationElem: bar : ,
// CHECK-NEXT: | | | `-VoidPlaceholder:
// CHECK-NEXT: | | `-DataDestructurationElem: foo :
// CHECK-NEXT: | |   `-VoidPlaceholder:
// CHECK-NEXT: | `-ExprIntegerNumber: 2
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-DataDestructuration: {}
// CHECK-NEXT: | | |-DataDestructurationElem: bar : ,
// CHECK-NEXT: | | | `-DataDestructuration: {}
// CHECK-NEXT: | | |   `-DataDestructurationElem: baz :
// CHECK-NEXT: | | |     `-ExprIntegerNumber: 2
// CHECK-NEXT: | | `-DataDestructurationElem: foo :
// CHECK-NEXT: | |   `-VoidPlaceholder:
// CHECK-NEXT: | `-ExprIntegerNumber: 3
// CHECK-NEXT: `-ExprMatchCase: ->
// CHECK-NEXT:   |-Otherwise:
// CHECK-NEXT:   `-ExprIntegerNumber: 10
