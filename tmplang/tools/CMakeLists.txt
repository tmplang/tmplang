add_llvm_executable(tmplangc
  tmplangc.cpp
)

target_link_libraries(tmplangc
  PRIVATE
  TmplangCLI

  LLVMSupport
  LLVMOption
)
