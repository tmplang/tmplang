// RUN: not tmplangc %s 2>&1 | FileCheck %s 

data =
  a : i32;

// CHECK:error: missing identifier on data declaration
// CHECK:data <data_decl_identifier> =
