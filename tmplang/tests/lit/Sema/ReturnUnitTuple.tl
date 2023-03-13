// RUN: not tmplangc %s --max-phase compilation 2>&1

fn foo -> (()) {}

// CHECK: subprogram with return type different to unit does not return in all possible paths
