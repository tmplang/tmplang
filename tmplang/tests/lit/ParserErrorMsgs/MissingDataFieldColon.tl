// RUN: not tmplangc %s 2>&1 | FileCheck %s 

data Dummy =
  a  i32;

// CHECK:error: missing colon between identifier and type on data field
// CHECK:a  : i32;