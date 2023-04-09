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
#include <tmplang/Diagnostics/Diagnostic.h>
#include <tmplang/Lexer/Lexer.h>
#include <tmplang/Lowering/HIRBuilder.h>
#include <tmplang/Lowering/Sema/Sema.h>
#include <tmplang/Parser/Parser.h>
#include <tmplang/Support/FileManager.h>
#include <tmplang/Support/SourceManager.h>

#include <tmplang/Lowering/Dialect/HIR/Traits.h>

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

  auto ctx = std::make_unique<mlir::MLIRContext>();
  // Load the required dialects (mlir::BuiltinDialect is loaded by default)
  ctx->loadDialect<TmplangHIRDialect>();

  auto tuOp = tmplang::LowerToHIR(*srcCompilationUnit, *ctx, *sm);
  if (!tuOp) {
    printer.errs() << "Lowering to HIR failed\n";
    return 1;
  }

  mlir::OpPrintingFlags flags;
  flags.assumeVerified();
  tuOp->print(llvm::dbgs(), flags);

  if (!tmplang::Sema(tuOp, *sm)) {
    printer.errs() << "Sema failed\n";
    return 1;
  }

  tuOp->print(llvm::dbgs(), flags);

  return 0;
}
