// RUN: not tmplangc %s 2>&1 | FileCheck %s 

proc foo { ret }

// CHECK:error: missing expression after ret keyword
