// RUN: tmplangc --dump-mlir=lower %s 2>&1 | FileCheck %s

fn foo -> (i32, i32, (i32, i32), ()) {
  ret (1, 2, (3, 4), ());
}

//      CHECK:tmplang.return(%6) -> tuple<i32, i32, tuple<i32, i32>, tuple<>>
