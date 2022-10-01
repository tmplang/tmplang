#include <tmplang/Tree/Source/CompilationUnit.h>

using namespace tmplang;
using namespace tmplang::source;

SourceLocation CompilationUnit::getBeginLoc() const {
  if (FunctionDeclarations.empty()) {
    return SourceLocation{};
  }
  return FunctionDeclarations.front().getBeginLoc();
}

SourceLocation CompilationUnit::getEndLoc() const {
  if (FunctionDeclarations.empty()) {
    return SourceLocation{};
  }
  return FunctionDeclarations.back().getEndLoc();
}
