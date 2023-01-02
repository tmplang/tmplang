// RUN: not tmplangc %s 2>&1 | FileCheck %s

fn foo -> i32 {}

// CHECK:error: subprogram with return type different to unit does not return in all possible paths
