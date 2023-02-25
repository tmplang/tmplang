set(TMPLANG_CONVERSION_DIR
    ${LLVM_MAIN_SRC_DIR}/../tmplang/include/tmplang/Lowering/Conversion)

add_subdirectory(TmplangToArith)
add_subdirectory(TmplangToFunc)
add_subdirectory(TmplangToLLVM)
