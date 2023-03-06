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

void SymbolicScope::dump() const {
  constexpr const char *headerFmt = "Scope addr: {0}\n";

  llvm::dbgs() << llvm::formatv(headerFmt, this);

  for (const Symbol &sym : getSymbols()) {
    constexpr const char *fmt = "  Sym:\n"
                                "  |-Id: {0}\n"
                                "  |-Type: {1}\n"
                                "  `-Kind: {2}\n";

    std::string str;
    llvm::raw_string_ostream rso(str);
    sym.getType().print(rso);

    llvm::formatv(fmt, sym.getId(), rso.str(), ToString(sym.getKind()))
        .format(llvm::dbgs());
  }
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

void Symbol::print(llvm::raw_ostream &out) const {
  getType().print(out);
  out << " : " << getId() << ", kind: " << ToString(this->getKind()) << "\n";
}

void Symbol::dump() const {
  print(llvm::dbgs());
}

bool Symbol::operator==(const Symbol &otherSym) const {
  return otherSym.Id == Id && otherSym.Kind == Kind;
}

Symbol::Symbol(const SymbolicScope &parentScope, const HIRContext &ctx)
    : Kind(SymbolKind::Unresolved), Ty(TupleType::getUnit(ctx)) {}
