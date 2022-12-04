proc foo: () {}

// RUN: not tmplangc %s --dump-src=simple 2>&1 | FileCheck %s

// CHECK:ParamDecl: () Err![<parameter_id>]
