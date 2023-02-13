data Dummy =
 foo : i32;

proc foo: (i32, Dummy) a -> i32 {
  ret match a {
    (1, a) -> 1,
    (a, b) -> a,
    (a, {foo : b}) -> b,
    (a, {foo : _}) -> 1,
    a -> 4,
    _ -> 5,
    otherwise -> 6
  };
} 

// RUN: tmplangc %s --dump-hir=simple 2>&1 | FileCheck %s

//      CHECK: ExprMatch: i32
// CHECK-NEXT: |-ExprVarRef: a (i32, Dummy{...})
// CHECK-NEXT: |-ExprMatchCase: i32
// CHECK-NEXT: | |-AggregateDestructuration: (i32, Dummy{...})
// CHECK-NEXT: | | |-AggregateDestructurationElem: i32
// CHECK-NEXT: | | | `-ExprIntegerNumber: i32 : 1
// CHECK-NEXT: | | `-AggregateDestructurationElem: Dummy{...}
// CHECK-NEXT: | |   `-PlaceholderDecl: Dummy{...} : a
// CHECK-NEXT: | `-ExprIntegerNumber: i32 : 1
// CHECK-NEXT: |-ExprMatchCase: i32
// CHECK-NEXT: | |-AggregateDestructuration: (i32, Dummy{...})
// CHECK-NEXT: | | |-AggregateDestructurationElem: i32
// CHECK-NEXT: | | | `-PlaceholderDecl: i32 : a
// CHECK-NEXT: | | `-AggregateDestructurationElem: Dummy{...}
// CHECK-NEXT: | |   `-PlaceholderDecl: Dummy{...} : b
// CHECK-NEXT: | `-ExprVarRef: a i32
// CHECK-NEXT: |-ExprMatchCase: i32
// CHECK-NEXT: | |-AggregateDestructuration: (i32, Dummy{...})
// CHECK-NEXT: | | |-AggregateDestructurationElem: i32
// CHECK-NEXT: | | | `-PlaceholderDecl: i32 : a
// CHECK-NEXT: | | `-AggregateDestructurationElem: Dummy{...}
// CHECK-NEXT: | |   `-AggregateDestructuration: Dummy{...}
// CHECK-NEXT: | |     `-AggregateDestructurationElem: i32
// CHECK-NEXT: | |       `-PlaceholderDecl: i32 : b
// CHECK-NEXT: | `-ExprVarRef: b i32
// CHECK-NEXT: |-ExprMatchCase: i32
// CHECK-NEXT: | |-AggregateDestructuration: (i32, Dummy{...})
// CHECK-NEXT: | | |-AggregateDestructurationElem: i32
// CHECK-NEXT: | | | `-PlaceholderDecl: i32 : a
// CHECK-NEXT: | | `-AggregateDestructurationElem: Dummy{...}
// CHECK-NEXT: | |   `-AggregateDestructuration: Dummy{...}
// CHECK-NEXT: | |     `-AggregateDestructurationElem: i32
// CHECK-NEXT: | `-ExprIntegerNumber: i32 : 1
// CHECK-NEXT: |-ExprMatchCase: i32
// CHECK-NEXT: | |-PlaceholderDecl: (i32, Dummy{...}) : a
// CHECK-NEXT: | `-ExprIntegerNumber: i32 : 4
// CHECK-NEXT: |-ExprMatchCase: i32
// CHECK-NEXT: | `-ExprIntegerNumber: i32 : 5
// CHECK-NEXT: `-ExprMatchCase: i32
// CHECK-NEXT:   `-ExprIntegerNumber: i32 : 6
