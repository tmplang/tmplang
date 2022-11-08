// RUN: not tmplangc %s 2>&1 | FileCheck %s

fn foo -> i32 {}

// CHECK:Sema failed!
