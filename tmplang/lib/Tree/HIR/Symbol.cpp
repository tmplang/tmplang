#include <tmplang/Tree/HIR/Symbol.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Tree/HIR/Types.h>

#include <cassert>

using namespace tmplang;
using namespace tmplang::hir;

bool SymbolicScope::isGlobalScope() const { return ParentScope == nullptr; }

const SymbolicScope &SymbolicScope::getContainingScope() const {
  assert(!isGlobalScope() && "GlobalScope does not have containing scope");
  return *ParentScope;
}

bool SymbolicScope::containsSymbol(const Symbol &sym) const {
  return llvm::find(Symbols, sym) != Symbols.end();
}

bool SymbolicScope::containsSymbol(SymbolKind kind, llvm::StringRef id) const {
  return llvm::find_if(Symbols, [=](const Symbol &sym) {
           return sym.getKind() == kind && sym.getId() == id;
         }) != Symbols.end();
}

Symbol &SymbolicScope::addSymbol(Symbol &sym) {
  return Symbols.emplace_back(sym);
}

void dumpImpl(const SymbolicScope &scope, bool recursively,
              unsigned level = 0) {
  auto *parentScope =
      scope.isGlobalScope() ? nullptr : &scope.getContainingScope();

  const unsigned numSpaces = level * 2;
  const std::string spacesStr = std::string(numSpaces, ' ');
  constexpr const char *headerFmt = "{0}Scope addr: {1}\n"
                                    "{0}`-> Parent scope: {2}\n";

  llvm::dbgs() << llvm::formatv(headerFmt, spacesStr, &scope, parentScope);

  for (const Symbol &sym : scope.getSymbols()) {
    constexpr const char *fmt = "{0}  Sym:\n"
                                "{0}  |-Id: {1}\n"
                                "{0}  |-Type: {2}\n"
                                "{0}  `-Kind: {3}\n";

    std::string str;
    llvm::raw_string_ostream rso(str);
    sym.getType().print(rso);

    llvm::formatv(fmt, spacesStr, sym.getId(), rso.str(),
                  ToString(sym.getKind()))
        .format(llvm::dbgs());
  }

  llvm::dbgs() << "\n";

  if (recursively && !scope.isGlobalScope()) {
    dumpImpl(scope.getContainingScope(), recursively, level + 1);
  }
}

void SymbolicScope::dump(bool recursively) const {
  dumpImpl(*this, recursively);
}

llvm::StringLiteral tmplang::hir::ToString(SymbolKind kind) {
  switch (kind) {
  case SymbolKind::Unresolved:
    return "Unknown";
  case SymbolKind::Subprogram:
    return "Subprogram";
  case SymbolKind::ParamOrVarDecl:
    return "ParamOrVarDecl";
  }
  llvm_unreachable("All cases are covered");
}

bool Symbol::operator==(const Symbol &otherSym) const {
  return otherSym.Id == Id && otherSym.Kind == Kind;
}

Symbol::Symbol(const SymbolicScope &parentScope, const HIRContext &ctx)
    : Kind(SymbolKind::Unresolved), Ty(TupleType::getUnit(ctx)) {}
