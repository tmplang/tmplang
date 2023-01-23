// RUN: not tmplangc %s --dump-hir=simple 2>&1

data Dummy =
  a : i32,
  b : (i32, i32);

data Dummy =
  a : Dummy;
