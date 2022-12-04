proc foo

// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

// CHECK:FuncDecl: proc foo Err![{] Err![}]
