// RUN: not tmplangc %s --dump-hir=simple 2>&1

data Dummy =
  a : i32,
  a : (i32, i32);
