add_tmplang_unittest(tmplangParserTests
    ParserTest.cpp
)

target_link_libraries(tmplangParserTests
  PRIVATE
  TmplangParser
)
