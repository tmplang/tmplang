#include <tmplang/Codegen/Codegen.h>

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/SubtargetFeature.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Target/TargetLoweringObjectFile.h>
#include <llvm/Target/TargetMachine.h>
#include <tmplang/ADT/LLVM.h>

using namespace tmplang;

static llvm::SubtargetFeatures GetSubtargetFeatures() {
  llvm::StringMap<bool> hostFeatures;
  llvm::SubtargetFeatures features;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
    for (auto &f : hostFeatures) {
      features.AddFeature(f.first(), f.second);
    }
  }
  return features;
}

CodegenResult tmplang::Codegen(llvm::Module &mod, StringRef outFile) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  std::string err;
  llvm::Triple targetTriple(llvm::sys::getDefaultTargetTriple());
  auto *theTarget = llvm::TargetRegistry::lookupTarget("", targetTriple, err);
  if (!theTarget) {
    return CodegenResult::TargetTripleNotSupported;
  }
  llvm::TargetOptions targetOpts;
  llvm::SubtargetFeatures features = GetSubtargetFeatures();
  std::unique_ptr<llvm::TargetMachine> targetMachine(
      theTarget->createTargetMachine(targetTriple.getTriple(),
                                     llvm::sys::getHostCPUName(),
                                     features.getString(), targetOpts, None,
                                     None, llvm::CodeGenOpt::Default));
  if (!targetMachine) {
    return CodegenResult::TargetMachineNotFound;
  }

  // This probably should be set in the MLIR module and inherited here
  mod.setDataLayout(targetMachine->createDataLayout());

  // Open the file.
  std::error_code ec;
  auto fileOut = std::make_unique<llvm::ToolOutputFile>(outFile, ec,
                                                        llvm::sys::fs::OF_None);
  if (ec) {
    return CodegenResult::FilesystemErrorCreatingOutput;
  }

  llvm::LLVMTargetMachine &LLVMTM =
      static_cast<llvm::LLVMTargetMachine &>(*targetMachine);
  auto *MMIWP = new llvm::MachineModuleInfoWrapperPass(&LLVMTM);

  llvm::legacy::PassManager passManager;
  llvm::TargetLibraryInfoImpl TLII(targetTriple);
  passManager.add(new llvm::TargetLibraryInfoWrapperPass(TLII));
  if (targetMachine->addPassesToEmitFile(
          passManager, fileOut->os(), /*split dwarf out stream*/ nullptr,
          llvm::CGFT_ObjectFile, /*verify*/ false, MMIWP)) {
    return CodegenResult::FileTypeNotSupported;
  }

  targetMachine->getObjFileLowering()->Initialize(MMIWP->getMMI().getContext(),
                                                  *targetMachine);

  passManager.run(mod);

  fileOut->keep();
  return CodegenResult::Ok;
}
