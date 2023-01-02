// RUN: not tmplangc %s 2>&1 | FileCheck %s

fn foo -> (i32, i32) { ret (1, ()); }

// CHECK:error: mismatch between returned expression type and subprogram return type
