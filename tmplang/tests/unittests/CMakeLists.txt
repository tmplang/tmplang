add_custom_target(tmplangUnitTests)
set_target_properties(tmplangUnitTests PROPERTIES FOLDER "Tests")

set(test_root_dir ${CMAKE_CURRENT_LIST_DIR})

function(add_tmplang_unittest test_dirname)
  add_unittest(tmplangUnitTests ${test_dirname} ${ARGN})
  # Using CMAKE_CURRENT_LIST_DIR here would yield the callee's folder
  target_include_directories(${test_dirname} PUBLIC ${test_root_dir}/common)
  target_compile_definitions(${test_dirname} PRIVATE TMPLANG_RESOURCE_PATH="${LLVM_LIBRARY_OUTPUT_INTDIR}")
endfunction()

add_subdirectory(lexer)
add_subdirectory(Parser)
add_subdirectory(HIR)
