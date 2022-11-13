#include <llvm/Option/ArgList.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/CLI/Arguments.h>
#include <tmplang/CLI/CLPrinter.h>
#include <tmplang/Lexer/Lexer.h>
#include <tmplang/Parser/Parser.h>
#include <tmplang/Sema/Sema.h>
#include <tmplang/Support/FileManager.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <memory>

using namespace tmplang;

static llvm::Optional<hir::Node::PrintConfig>
ParseDumpHIRArg(llvm::opt::Arg &arg, CLPrinter &out) {
  if (arg.getNumValues() == 0) {
    out.errs() << "At least one value of the followings is required: "
                  "'color|addr|loc|all|simple'\n";
    return llvm::None;
  }

  hir::Node::PrintConfig printCfg = hir::Node::PrintConfig::None;
  for (llvm::StringRef option : arg.getValues()) {
    llvm::Optional<tmplang::hir::Node::PrintConfig> parsedOption =
        llvm::StringSwitch<llvm::Optional<hir::Node::PrintConfig>>(option)
            .Case("color", hir::Node::PrintConfig::Color)
            .Case("addr", hir::Node::PrintConfig::Address)
            .Case("loc", hir::Node::PrintConfig::SourceLocation)
            .Case("simple", hir::Node::PrintConfig::None)
            .Case("all", hir::Node::PrintConfig::All)
            .Default(llvm::None);

    if (!parsedOption) {
      out.errs() << "Invalid value '" << option << "' for flag "
                 << arg.getSpelling() << "\n";
      return llvm::None;
    }

    printCfg |= *parsedOption;
  }

  return printCfg;
}

static bool DumpHIR(llvm::opt::Arg &arg, CLPrinter &out,
                    const hir::CompilationUnit &compUnit,
                    const SourceManager &sm) {
  llvm::Optional<hir::Node::PrintConfig> cfg = ParseDumpHIRArg(arg, out);
  if (!cfg) {
    // Errors already reported
    return 1;
  }

  compUnit.dump(sm, *cfg);
  return 0;
}

int main(int argc, const char *argv[]) {
  llvm::InitLLVM llvm(argc, argv);

  // Let's avoid vulnerabilities exploits
  if (argc < 1) {
    return -1;
  }

  CLPrinter printer(llvm::outs(), llvm::errs(), argv[0]);

  auto compilerArgs = llvm::makeArrayRef(argv + 1, argv + argc);
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

  Lexer lexer(targetFileEntry->Content->getBuffer());
  auto srcCompilationUnit = Parse(lexer);
  if (!srcCompilationUnit) {
    // TODO: Add proper message diagnostic handling
    printer.errs() << "There was a problem parsing the file\n";
    return 1;
  }

  hir::HIRContext ctx;
  auto hirCompilationUnit = hir::buildHIR(*srcCompilationUnit, ctx);
  if (!hirCompilationUnit) {
    // TODO: Add proper message diagnostic handling
    printer.errs() << "There was a problem building the HIR for the file\n";
    return 1;
  }

  auto sm = std::make_unique<SourceManager>(*targetFileEntry);
  if (auto *dumpHIRArg = parsedCompilerArgs->getLastArg(OPT_dump_hir)) {
    return DumpHIR(*dumpHIRArg, printer, *hirCompilationUnit, *sm);
  }

  if (!tmplang::Sema(*hirCompilationUnit)) {
    printer.errs() << "Sema failed!\n";
    return 1;
  }

  return 0;
}
