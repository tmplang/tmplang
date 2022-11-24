// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo }

// CHECK:error: missing left key brace
// CHECK:proc foo { }
