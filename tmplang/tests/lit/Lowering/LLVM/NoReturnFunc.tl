// RUN: tmplangc --dump-mlir=llvm %s 2>&1 | FileCheck %s

fn foo {}

//      CHECK:define {} @foo() !dbg !3 {
// CHECK-NEXT:  %1 = alloca {}, i64 1, align 8
// CHECK-NEXT:  %2 = load {}, ptr %1, align 1
// CHECK-NEXT:  ret {} %2
// CHECK-NEXT:}
