add_llvm_executable(tmplang-opt
  tmplang-opt.cpp
)

llvm_update_compile_flags(tmplang-opt)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(tmplang-opt PRIVATE
  TmplangMLIRDialect
  TmplangToArith
  TmplangToFunc
  TmplangToLLVM

  MLIRLLVMDialect
  MLIROptLib
)
