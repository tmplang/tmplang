#ifndef TMPLANG_TESTS_UNITTESTS_COMMON_PARSER_H
#define TMPLANG_TESTS_UNITTESTS_COMMON_PARSER_H

#include <llvm/Support/VirtualFileSystem.h>
#include <tmplang/Parser/Parser.h>
#include <tmplang/Support/FileManager.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/HIRBuilder.h>
#include <tmplang/Tree/Source/CompilationUnit.h>

inline llvm::Optional<tmplang::source::CompilationUnit>
CleanParse(llvm::StringRef code) {
  auto inMemoryFileSystem = std::make_unique<llvm::vfs::InMemoryFileSystem>();

  const char *fileName = "./test";
  inMemoryFileSystem->addFile(fileName, 0,
                              llvm::MemoryBuffer::getMemBuffer(code));

  tmplang::FileManager fm(std::move(inMemoryFileSystem));
  const tmplang::TargetFileEntry *tfe = fm.findOrOpenTargetFile(fileName);
  assert(tfe);

  const tmplang::SourceManager sm(*tfe);
  tmplang::Lexer lex(code);

  return Parse(lex, llvm::nulls(), sm);
}

#endif // TMPLANG_TESTS_UNITTESTS_COMMON_PARSER_H
