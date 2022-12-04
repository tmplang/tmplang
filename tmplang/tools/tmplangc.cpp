#include "tmplang/Tree/Source/Node.h"
#include <llvm/Option/ArgList.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/CLI/Arguments.h>
#include <tmplang/CLI/CLPrinter.h>
#include <tmplang/Diagnostics/Diagnostic.h>
#include <tmplang/Lexer/Lexer.h>
#include <tmplang/Parser/Parser.h>
#include <tmplang/Sema/Sema.h>
#include <tmplang/Support/FileManager.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <memory>

using namespace tmplang;

template <typename PrintConfig_t>
static llvm::Optional<PrintConfig_t> ParseDumpArg(llvm::opt::Arg &arg,
                                                  CLPrinter &out) {
  if (arg.getNumValues() == 0) {
    out.errs() << "At least one value of the followings is required: "
                  "'color|addr|loc|all|simple'\n";
    return llvm::None;
  }

  PrintConfig_t printCfg = PrintConfig_t::None;
  for (llvm::StringRef option : arg.getValues()) {
    llvm::Optional<PrintConfig_t> parsedOption =
        llvm::StringSwitch<llvm::Optional<PrintConfig_t>>(option)
            .Case("color", PrintConfig_t::Color)
            .Case("addr", PrintConfig_t::Address)
            .Case("loc", PrintConfig_t::SourceLocation)
            .Case("simple", PrintConfig_t::None)
            .Case("all", PrintConfig_t::All)
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

static bool DumpSrc(llvm::opt::Arg &arg, CLPrinter &out,
                    const source::CompilationUnit &compUnit,
                    const SourceManager &sm) {
  auto cfg = ParseDumpArg<source::Node::PrintConfig>(arg, out);
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
  auto cfg = ParseDumpArg<hir::Node::PrintConfig>(arg, out);
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

  auto sm = std::make_unique<SourceManager>(*targetFileEntry);

  tmplang::diagnostic_ostream diagOuts(
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

  if (!tmplang::Sema(*hirCompilationUnit)) {
    printer.errs() << "Sema failed!\n";
    return 1;
  }

  return srcCompilationUnit->didRecoverFromAnError();
}
