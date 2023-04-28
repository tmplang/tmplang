#ifndef TMPLANG_BASE
#define TMPLANG_BASE

include "mlir/IR/DialectBase.td"

// Provide a definition of the 'tmplang' dialect in the ODS framework so that we
// can define our operations.
def Tmplang_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "tmplang";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Tmplang language";

  let extraClassDeclaration = [{
    void registerOps();
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;

  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // TMPLANG_BASE
