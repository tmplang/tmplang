#include <llvm/Option/ArgList.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <tmplang/ADT/LLVM.h>
#include <tmplang/CLI/Arguments.h>
#include <tmplang/CLI/CLPrinter.h>
#include <tmplang/Codegen/Codegen.h>
#include <tmplang/Diagnostics/Diagnostic.h>
#include <tmplang/Lexer/Lexer.h>
#include <tmplang/Lowering/Lowering.h>
#include <tmplang/Parser/Parser.h>
#include <tmplang/Sema/Sema.h>
#include <tmplang/Support/FileManager.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <memory>

using namespace tmplang;

template <typename PrintConfig_t>
static std::optional<PrintConfig_t> ParseTreeDumpArg(llvm::opt::Arg &arg,
                                                     CLPrinter &out) {
  if (arg.getNumValues() == 0) {
    out.errs() << "At least one value of the followings is required: "
                  "'color|addr|loc|all|simple'\n";
    return nullopt;
  }

  PrintConfig_t printCfg = PrintConfig_t::None;
  for (StringRef option : arg.getValues()) {
    std::optional<PrintConfig_t> parsedOption =
        StringSwitch<std::optional<PrintConfig_t>>(option)
            .Case("color", PrintConfig_t::Color)
            .Case("addr", PrintConfig_t::Address)
            .Case("loc", PrintConfig_t::SourceLocation)
            .Case("simple", PrintConfig_t::None)
            .Case("all", PrintConfig_t::All)
            .Default(nullopt);

    if (!parsedOption) {
      out.errs() << "Invalid value '" << option << "' for flag "
                 << arg.getSpelling() << "\n";
      return nullopt;
    }

    printCfg |= *parsedOption;
  }

  return printCfg;
}

static bool DumpSrc(llvm::opt::Arg &arg, CLPrinter &out,
                    const source::CompilationUnit &compUnit,
                    const SourceManager &sm) {
  auto cfg = ParseTreeDumpArg<source::Node::PrintConfig>(arg, out);
  if (!cfg) {
    // Errors already reported
    return 1;
  }

  compUnit.dump(sm, *cfg);
  return compUnit.didRecoverFromAnError();
}

static bool DumpHIR(llvm::opt::Arg &arg, CLPrinter &out,
                    const hir::CompilationUnit &compUnit,
                    const SourceManager &sm) {
  auto cfg = ParseTreeDumpArg<hir::Node::PrintConfig>(arg, out);
  if (!cfg) {
    // Errors already reported
    return 1;
  }

  compUnit.dump(sm, *cfg);
  return 0;
}

static std::optional<MLIRPrintingOpsCfg> ParseDumpMLIRArg(llvm::opt::Arg &arg,
                                                          CLPrinter &out) {
  constexpr StringLiteral missingDumpOptionMsg =
      "At least one value of the followings is required: "
      "'lower|opt|trans|llvm'\n";

  if (arg.getNumValues() == 0) {
    out.errs() << missingDumpOptionMsg;
    return nullopt;
  }

  MLIRPrintingOpsCfg printCfg = MLIRPrintingOpsCfg::None;
  for (StringRef option : arg.getValues()) {
    std::optional<MLIRPrintingOpsCfg> parsedOption =
        StringSwitch<std::optional<MLIRPrintingOpsCfg>>(option)
            .Case("lower", MLIRPrintingOpsCfg::Lowering)
            .Case("opt", MLIRPrintingOpsCfg::Optimization)
            .Case("trans", MLIRPrintingOpsCfg::Translation)
            .Case("llvm", MLIRPrintingOpsCfg::LLVM)
            .Case("loc", MLIRPrintingOpsCfg::Location)
            .Case("all", MLIRPrintingOpsCfg::All)
            .Default(nullopt);

    if (!parsedOption) {
      out.errs() << "Invalid value '" << option << "' for flag "
                 << arg.getSpelling() << "\n";
      return nullopt;
    }

    printCfg |= *parsedOption;
  }

  // Location is a complemetary option, by itself it does not count
  if (!static_cast<bool>((printCfg & ~MLIRPrintingOpsCfg::Location))) {
    out.errs() << missingDumpOptionMsg;
    return nullopt;
  }

  return printCfg;
}

static std::string GetInputAsObjectFile(StringRef input) {
  return (input.rsplit(".").first + ".o").str();
}

int main(int argc, const char *argv[]) {
  llvm::InitLLVM llvm(argc, argv);

  // Let's avoid vulnerabilities exploits
  if (argc < 1) {
    return -1;
  }

  CLPrinter printer(llvm::outs(), llvm::errs(), argv[0]);

  ArrayRef compilerArgs(argv + 1, argv + argc);
  std::unique_ptr<llvm::opt::InputArgList> parsedCompilerArgs =
      ParseArgs(compilerArgs, printer);
  if (!parsedCompilerArgs) {
    return 1;
  }

  if (HandleImmediateArgs(*parsedCompilerArgs, printer)) {
    return 0;
  }

  std::vector<std::string> inputs =
      parsedCompilerArgs->getAllArgValues(OPT_INPUT);
  const std::size_t nInputs = inputs.size();
  if (nInputs != 1) {
    printer.errs() << llvm::formatv(
        "expected exactly one input, got {0} inputs\n", nInputs);
    return 1;
  }

  auto fm =
      std::make_unique<FileManager>(llvm::vfs::createPhysicalFileSystem());
  assert(fm);

  const TargetFileEntry *targetFileEntry = fm->findOrOpenTargetFile(inputs[0]);
  if (!targetFileEntry) {
    printer.errs() << "could not open file: '" << inputs[0] << "'\n";
    return 1;
  }

  auto sm = std::make_unique<SourceManager>(*targetFileEntry);

  diagnostic_ostream diagOuts(
      parsedCompilerArgs->hasArg(OPT_color_diagnostics));

  Lexer lexer(targetFileEntry->Content->getBuffer());
  auto srcCompilationUnit = Parse(lexer, diagOuts, *sm);
  if (!srcCompilationUnit) {
    // TODO: Add proper message diagnostic handling
    printer.errs() << "There was a problem parsing the file\n";
    return 1;
  }

  if (auto *dumpSrcArg = parsedCompilerArgs->getLastArg(OPT_dump_src)) {
    return DumpSrc(*dumpSrcArg, printer, *srcCompilationUnit, *sm);
  }
  StringRef maximumPhase = parsedCompilerArgs->getLastArgValue(OPT_max_phase);
  if (maximumPhase == "syntax") {
    return 0;
  }

  hir::HIRContext ctx;
  auto hirCompilationUnit = hir::buildHIR(*srcCompilationUnit, ctx);
  if (!hirCompilationUnit) {
    // TODO: Add proper message diagnostic handling
    printer.errs() << "There was a problem building the HIR for the file\n";
    return 1;
  }

  if (auto *dumpHIRArg = parsedCompilerArgs->getLastArg(OPT_dump_hir)) {
    return DumpHIR(*dumpHIRArg, printer, *hirCompilationUnit, *sm);
  }

  if (!Sema(*hirCompilationUnit, *sm, diagOuts)) {
    printer.errs() << "Sema failed!\n";
    return 1;
  }

  // Stop here, if there was a recoverable error while parsing
  if (srcCompilationUnit->didRecoverFromAnError()) {
    return 1;
  }

  auto *dumpMLIRArg = parsedCompilerArgs->getLastArg(OPT_dump_mlir);
  std::optional<MLIRPrintingOpsCfg> mlirDumpCfg =
      dumpMLIRArg ? ParseDumpMLIRArg(*dumpMLIRArg, printer)
                  : MLIRPrintingOpsCfg::None;
  if (!mlirDumpCfg) {
    // Errors already reported
    return 1;
  }

  auto llvmCtx = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> mlirMod =
      Lower(*hirCompilationUnit, *llvmCtx, *sm, *mlirDumpCfg);
  if (!mlirMod) {
    printer.errs() << "MLIR lowering to LLVM failed!\n";
    return 1;
  }
  if (dumpMLIRArg || maximumPhase == "compilation") {
    return 0;
  }

  int result = 1;
  switch (Codegen(*mlirMod, parsedCompilerArgs->getLastArgValue(
                                OPT_output, GetInputAsObjectFile(inputs[0])))) {
  case CodegenResult::Ok:
    result = 0;
    break;
  case CodegenResult::TargetTripleNotSupported:
    printer.errs() << "Codegen target triple not supported\n";
    break;
  case CodegenResult::TargetMachineNotFound:
    printer.errs() << "Codegen target machine not found!\n";
    break;
  case CodegenResult::FilesystemErrorCreatingOutput:
    printer.errs() << "Filesystem error creating output file\n";
    break;
  case CodegenResult::FileTypeNotSupported:
    printer.errs() << "target does not support generation of this file type";
    break;
  }

  return result;
}
