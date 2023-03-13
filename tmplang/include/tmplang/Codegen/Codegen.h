#ifndef TMPLANG_CODEGEN_CODEGEN_H
#define TMPLANG_CODEGEN_CODEGEN_H

#include <llvm/ADT/StringRef.h>

namespace llvm {
class Module;
} // namespace llvm

namespace tmplang {

// TODO: turn this into an llvm::ErrorInfo struct with messages
enum class CodegenResult {
  Ok,
  TargetTripleNotSupported,
  TargetMachineNotFound, // Not sure why this is
  FilesystemErrorCreatingOutput,
  FileTypeNotSupported,
};

CodegenResult Codegen(llvm::Module &mod, llvm::StringRef outFile);

} // namespace tmplang

#endif // TMPLANG_CODEGEN_CODEGEN_H
