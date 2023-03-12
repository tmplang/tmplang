#ifndef TMPLANG_LOWERING_TRANSFORMS_PASSES
#define TMPLANG_LOWERING_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def RemoveUnreachableOpsAfterReturn : Pass<"remove-ops-after-return", "tmplang::SubprogramOp"> {
  let summary = "Remove all operations that happen to be after a tmplang.return";
  let description = [{
    Returns all instructions in the same basic block, that are unreachable, after a
    tmplang.return operation.

    Before:

    ...
    tmplang.return (%0)
    %1 = tmplang.constant 1 : i32

    After:
    ...
    tmplang.return (%0)
  }];
  let constructor = "tmplang::createRemoveUnreachableOpsAfterReturnPass()";
  let dependentDialects = [
    "tmplang::TmplangDialect"
  ];
}

#endif // TMPLANG_LOWERING_TRANSFORMS_PASSES
