// RUN: not tmplangc %s 2>&1 | FileCheck %s 

foo {}

// XFAIL: *

// CHECK:error: missing subprogram classification on subprogram definition
// CHECK:{fn, proc} foo {}
