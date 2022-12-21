// RUN: not tmplangc %s 2>&1 | FileCheck %s 

fn foo { 5 5; }

// CHECK:error: missing semicolon
// CHECK:fn foo { 5 ; 5; }
