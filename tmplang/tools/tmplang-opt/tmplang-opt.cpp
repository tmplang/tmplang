#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <tmplang/Lowering/InitTmplang.h>

int main(int argc, char **argv) {
  tmplang::registerMLIRPassesForTmplang();
  tmplang::registerTmplangPasses();

  mlir::DialectRegistry registry;
  tmplang::registerDialects(registry);

  llvm::StringRef toolName = "Tmplang modular optimizer driver\n";
  return mlir::failed(MlirOptMain(argc, argv, toolName, registry,
                                  /*preloadDialectsInContext=*/false));
}
