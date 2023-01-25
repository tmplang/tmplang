// RUN: not tmplangc %s --dump-hir=simple 2>&1

fn foo {}
fn bar {
  foo;
}
