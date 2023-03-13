fn foo -> i32 {
  ret 7;
}

// RUN: tmplangc -o %t %s
// RUN: llvm-objdump -d %t | FileCheck %s

//      CHECK: <foo>:
// CHECK-NEXT:   movl $7, %eax
// CHECK-NEXT:   retq
