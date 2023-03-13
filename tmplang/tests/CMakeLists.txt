# Test runner infrastructure. This configures the test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit/lit.cfg.py
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unittests/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/unittests/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/unittests/lit.cfg.py
)

option(TMPLANG_TEST_USE_VG "Run tests under Valgrind" OFF)
if(TMPLANG_TEST_USE_VG)
  set(TMPLANG_TEST_EXTRA_ARGS ${TMPLANG_TEST_EXTRA_ARGS} "--vg")
endif ()

list(APPEND TMPLANG_TEST_DEPS
    FileCheck count not
    llvm-config
    llvm-objdump
    llvm-symbolizer

    tmplangc
    tmplangUnitTests
)

add_custom_target(tmplang-test-depends DEPENDS ${TMPLANG_TEST_DEPS})
set_target_properties(tmplang-test-depends PROPERTIES FOLDER "tmplang tests")

set(TMPLANG_TEST_PARAMS)

add_lit_testsuite(check-tmplang "Running the tmplang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}/lit ${CMAKE_CURRENT_BINARY_DIR}/unittests
  #LIT ${LLVM_LIT}
  PARAMS ${TMPLANG_TEST_PARAMS}
  DEPENDS ${TMPLANG_TEST_DEPS}
  ARGS ${TMPLANG_TEST_EXTRA_ARGS}
)
set_target_properties(check-tmplang PROPERTIES FOLDER "tmplang tests")

add_subdirectory(unittests)
