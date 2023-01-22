// RUN: not tmplangc %s 2>&1 | FileCheck %s 

data Dummy 
  a : i32;

// CHECK:error: missing mandatory '=' on data declaration
// CHECK:data Dummy =
