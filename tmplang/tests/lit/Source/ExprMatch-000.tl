// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

proc foo {
  match a {
    b -> 5,
    _ -> 4,
    1 -> 5,
    otherwise -> 10
  };
}

//      CHECK: ExprMatch: match { }
// CHECK-NEXT: |-ExprVarRef: a
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-PlaceholderDecl: b
// CHECK-NEXT: | `-ExprIntegerNumber: 5
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-VoidPlaceholder: 
// CHECK-NEXT: | `-ExprIntegerNumber: 4
// CHECK-NEXT: |-ExprMatchCase: ->
// CHECK-NEXT: | |-ExprIntegerNumber: 1
// CHECK-NEXT: | `-ExprIntegerNumber: 5
// CHECK-NEXT: `-ExprMatchCase: ->
// CHECK-NEXT:   |-Otherwise: 
// CHECK-NEXT:   `-ExprIntegerNumber: 10
