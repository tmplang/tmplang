fn -> () {}

// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

// CHECK:SubprogramDecl: fn Err![<subprogram_identifier>] -> ()
