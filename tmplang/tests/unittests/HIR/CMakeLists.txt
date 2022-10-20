add_tmplang_unittest(tmplangHIRTests
    HIRTest.cpp
)

target_link_libraries(tmplangHIRTests
  PRIVATE
  TmplangParser
  TmplangHIRTree
  TmplangSourceTree
)
