// RUN: not tmplangc %s 2>&1 | FileCheck %s 

? foo {}

// CHECK:error: found unexpected token
//      CHECK: 3 | ? foo {}
// CHECK-NEXT:   | ^
