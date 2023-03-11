add_llvm_executable(tmplangc
  tmplangc.cpp
)

target_link_libraries(tmplangc
  PRIVATE
  TmplangCLI
  TmplangLexer
  TmplangLowering
  TmplangParser
  TmplangSourceTree
  TmplangSema
  TmplangHIRTree
  TmplangSupport

  LLVMSupport
  LLVMOption
)

add_subdirectory(tmplang-opt)
