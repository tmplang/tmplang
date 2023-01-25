#include <tmplang/Tree/HIR/Symbol.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Tree/HIR/Types.h>

#include <cassert>

using namespace tmplang;
using namespace tmplang::hir;

bool SymbolicScope::containsSymbol(SymbolKind kind, llvm::StringRef id) const {
  return llvm::find_if(Symbols, [=](const Symbol &sym) {
           return sym.getKind() == kind && sym.getId() == id;
         }) != Symbols.end();
}

Symbol &SymbolicScope::addSymbol(Symbol &sym) {
  return Symbols.emplace_back(sym);
}

void dumpImpl(const SymbolicScope &scope, bool recursively) {
  constexpr const char *headerFmt = "Scope addr: {1}\n";

  llvm::dbgs() << llvm::formatv(headerFmt, &scope);

  for (const Symbol &sym : scope.getSymbols()) {
    constexpr const char *fmt = "  Sym:\n"
                                "  |-Id: {1}\n"
                                "  |-Type: {2}\n"
                                "  `-Kind: {3}\n";

    std::string str;
    llvm::raw_string_ostream rso(str);
    sym.getType().print(rso);

    llvm::formatv(fmt, sym.getId(), rso.str(), ToString(sym.getKind()))
        .format(llvm::dbgs());
  }
}

void SymbolicScope::dump(bool recursively) const {
  dumpImpl(*this, recursively);
}

llvm::StringLiteral tmplang::hir::ToString(SymbolKind kind) {
  switch (kind) {
  case SymbolKind::Unresolved:
    return "Unresolved";
  case SymbolKind::UnreferenciableDataFieldDecl:
    return "UnreferenciableDataFieldDecl";
  case SymbolKind::ReferenciableFromExprVarRef:
    return "ReferenciableFromExprVarRef";
  case SymbolKind::ReferenciableFromExprCall:
    return "ReferenciableFromExprCall";
  case SymbolKind::ReferenciableFromType:
    return "ReferenciableFromType";
  }
  llvm_unreachable("All cases are covered");
}

bool Symbol::operator==(const Symbol &otherSym) const {
  return otherSym.Id == Id && otherSym.Kind == Kind;
}

Symbol::Symbol(const SymbolicScope &parentScope, const HIRContext &ctx)
    : Kind(SymbolKind::Unresolved), Ty(TupleType::getUnit(ctx)) {}
