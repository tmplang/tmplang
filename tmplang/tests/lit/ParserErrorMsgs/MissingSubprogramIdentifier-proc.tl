// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc{}

// CHECK:error: missing subprogram identifier
// CHECK:proc <subprogram_identifier> {}
