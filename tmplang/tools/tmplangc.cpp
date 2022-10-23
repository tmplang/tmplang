#include <llvm/Option/ArgList.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/CLI/Arguments.h>
#include <tmplang/CLI/CLPrinter.h>
#include <tmplang/Lexer/Lexer.h>
#include <tmplang/Parser/Parser.h>
#include <tmplang/Tree/HIR/HIRBuilder.h>

using namespace tmplang;

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

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrStdIn =
      llvm::MemoryBuffer::getFileOrSTDIN(inputs[0], /*isText*/ true);
  if (std::error_code error = fileOrStdIn.getError()) {
    printer.errs() << error.message() << ": '" << inputs[0] << "'\n";
    return 1;
  }

  Lexer lexer(fileOrStdIn.get()->getBuffer());
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

  if (parsedCompilerArgs->hasArg(OPT_dump_hir)) {
    hirCompilationUnit->dump();
    return 0;
  }

  return 0;
}
