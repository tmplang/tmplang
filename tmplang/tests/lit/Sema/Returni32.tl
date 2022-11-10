// RUN: tmplangc %s --dump-hir=simple

fn foo0 -> i32 {}
fn foo1 -> (i32) {}
fn foo2 -> () {}
fn foo3 -> (((((((()))))))) {}
