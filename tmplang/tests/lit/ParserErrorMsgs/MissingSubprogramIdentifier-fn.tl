// RUN: not tmplangc %s 2>&1 | FileCheck %s 

fn{}

// CHECK:error: missing subprogram identifier
// CHECK:fn <subprogram_identifier> {}
