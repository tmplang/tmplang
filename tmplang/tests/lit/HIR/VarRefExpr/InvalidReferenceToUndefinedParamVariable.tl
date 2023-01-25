// RUN: not tmplangc %s --dump-hir=simple 2>&1

fn foo: a i32 {
  b;
}
