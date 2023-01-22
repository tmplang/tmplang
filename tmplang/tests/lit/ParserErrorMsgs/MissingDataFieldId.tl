// RUN: not tmplangc %s 2>&1 | FileCheck %s 

data Dummy =
   : i32;

// CHECK:error: missing identifier on data field
// CHECK:<data_field_identifier> : i32;
