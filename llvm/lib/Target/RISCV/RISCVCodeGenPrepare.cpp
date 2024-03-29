//===----- RISCVCodeGenPrepare.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a RISC-V specific version of CodeGenPrepare.
// It munges the code in the input function to better prepare it for
// SelectionDAG-based code generation. This works around limitations in it's
// basic-block-at-a-time approach.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-codegenprepare"
#define PASS_NAME "RISC-V CodeGenPrepare"

STATISTIC(NumZExtToSExt, "Number of SExt instructions converted to ZExt");

namespace {

class RISCVCodeGenPrepare : public FunctionPass,
                            public InstVisitor<RISCVCodeGenPrepare, bool> {
  const DataLayout *DL;
  const RISCVSubtarget *ST;

public:
  static char ID;

  RISCVCodeGenPrepare() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return PASS_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }

  bool visitInstruction(Instruction &I) { return false; }
  bool visitZExtInst(ZExtInst &I);
  bool visitAnd(BinaryOperator &BO);
};

} // end anonymous namespace

bool RISCVCodeGenPrepare::visitZExtInst(ZExtInst &ZExt) {
  if (!ST->is64Bit())
    return false;

  Value *Src = ZExt.getOperand(0);

  // We only care about ZExt from i32 to i64.
  if (!ZExt.getType()->isIntegerTy(64) || !Src->getType()->isIntegerTy(32))
    return false;

  // Look for an opportunity to replace (i64 (zext (i32 X))) with a sext if we
  // can determine that the sign bit of X is zero via a dominating condition.
  // This often occurs with widened induction variables.
  if (isImpliedByDomCondition(ICmpInst::ICMP_SGE, Src,
                              Constant::getNullValue(Src->getType()), ZExt,
                              *DL).value_or(false)) {
    auto *SExt = new SExtInst(Src, ZExt->getType(), "", ZExt);
    SExt->takeName(ZExt);
    SExt->setDebugLoc(ZExt->getDebugLoc());

    ZExt.replaceAllUsesWith(SExt);
    ZExt.eraseFromParent();
    ++NumZExtToSExt;
    return true;
  }

  // Convert (zext (abs(i32 X, i1 1))) -> (sext (abs(i32 X, i1 1))). If abs of
  // INT_MIN is poison, the sign bit is zero.
  using namespace PatternMatch;
  if (match(Src, m_Intrinsic<Intrinsic::abs>(m_Value(), m_One()))) {
    auto *SExt = new SExtInst(Src, ZExt.getType(), "", &ZExt);
    SExt->takeName(&ZExt);
    SExt->setDebugLoc(ZExt.getDebugLoc());

    ZExt.replaceAllUsesWith(SExt);
    ZExt.eraseFromParent();
    ++NumZExtToSExt;
    return true;
  }

  return false;
}

// Try to optimize (i64 (and (zext/sext (i32 X), C1))) if C1 has bit 31 set,
// but bits 63:32 are zero. If we can prove that bit 31 of X is 0, we can fill
// the upper 32 bits with ones. A separate transform will turn (zext X) into
// (sext X) for the same condition.
bool RISCVCodeGenPrepare::visitAnd(BinaryOperator &BO) {
  if (!ST->is64Bit())
    return false;

  if (!BO.getType()->isIntegerTy(64))
    return false;

  // Left hand side should be sext or zext.
  Instruction *LHS = dyn_cast<Instruction>(BO.getOperand(0));
  if (!LHS || (!isa<SExtInst>(LHS) && !isa<ZExtInst>(LHS)))
    return false;

  Value *LHSSrc = LHS->getOperand(0);
  if (!LHSSrc->getType()->isIntegerTy(32))
    return false;

  // Right hand side should be a constant.
  Value *RHS = BO.getOperand(1);

  auto *CI = dyn_cast<ConstantInt>(RHS);
  if (!CI)
    return false;
  uint64_t C = CI->getZExtValue();

  // Look for constants that fit in 32 bits but not simm12, and can be made
  // into simm12 by sign extending bit 31. This will allow use of ANDI.
  // TODO: Is worth making simm32?
  if (!isUInt<32>(C) || isInt<12>(C) || !isInt<12>(SignExtend64<32>(C)))
    return false;

  // If we can determine the sign bit of the input is 0, we can replace the
  // And mask constant.
  if (!isImpliedByDomCondition(ICmpInst::ICMP_SGE, LHSSrc,
                               Constant::getNullValue(LHSSrc->getType()),
                               LHS, *DL).value_or(false))
    return false;

  // Sign extend the constant and replace the And operand.
  C = SignExtend64<32>(C);
  BO.setOperand(1, ConstantInt::get(LHS->getType(), C));

  return true;
}

bool RISCVCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();
  auto &TM = TPC.getTM<RISCVTargetMachine>();
  ST = &TM.getSubtarget<RISCVSubtarget>(F);

  DL = &F.getParent()->getDataLayout();

  bool MadeChange = false;
  for (auto &BB : F)
    for (Instruction &I : llvm::make_early_inc_range(BB))
      MadeChange |= visit(I);

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(RISCVCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVCodeGenPrepare, DEBUG_TYPE, PASS_NAME, false, false)

char RISCVCodeGenPrepare::ID = 0;

FunctionPass *llvm::createRISCVCodeGenPreparePass() {
  return new RISCVCodeGenPrepare();
}
