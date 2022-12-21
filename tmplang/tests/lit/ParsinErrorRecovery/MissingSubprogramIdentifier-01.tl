proc -> () {}

// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

// CHECK:FunctionDecl: proc Err![<subprogram_identifier>] -> ()
