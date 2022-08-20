add_tmplang_unittest(tmplangLexerTests
    BasicLexer.cpp
)

target_link_libraries(tmplangLexerTests
  PRIVATE
  TmplangLexer
)
