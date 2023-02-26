// RUN: tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

union BinaryExpr =
  number   (value: i32),
  binaryOp (lhs: ArithmeticExpr, op: Operation, rhs: ArithmeticExpr);

//      CHECK:UnionDecl: union BinaryExpr = ;
// CHECK-NEXT:|-UnionAlternativeDecl: number ( )
// CHECK-NEXT:| `-UnionAlternativeFieldDecl: value : i32
// CHECK-NEXT:`-UnionAlternativeDecl: binaryOp ( )
// CHECK-NEXT:  |-UnionAlternativeFieldDecl: lhs : ArithmeticExpr ,
// CHECK-NEXT:  |-UnionAlternativeFieldDecl: op : Operation ,
// CHECK-NEXT:  `-UnionAlternativeFieldDecl: rhs : ArithmeticExpr
