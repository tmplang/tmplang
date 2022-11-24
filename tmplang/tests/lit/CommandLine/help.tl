// RUN: tmplangc --help | FileCheck %s

//      CHECK:Syntax: [[FILEPATH:.*]]tmplangc [options] <input>
//CHECK-EMPTY:
// CHECK-NEXT:Diagnostics options:
// CHECK-NEXT:  --color-diagnostics            prints colorful diagnostics
//CHECK-EMPTY:
// CHECK-NEXT:Dump options:
// CHECK-NEXT:  --dump-hir=                    multi options: color|addr|loc|all|simple
// CHECK-NEXT:                                 Default: all
//CHECK-EMPTY:
// CHECK-NEXT:Options:
// CHECK-NEXT:  -h [ --help ]                  print help
// CHECK-NEXT:  -v [ --version ]               print version
