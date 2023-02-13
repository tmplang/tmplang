#include <tmplang/Tree/HIR/Decls.h>

#include <tmplang/Tree/HIR/Exprs.h>
#include <tmplang/Tree/HIR/Symbol.h>

using namespace tmplang::hir;

StringRef Decl::getName() const { return Sym.getId(); }
const Type &Decl::getType() const { return Sym.getType(); }

const SubprogramType &SubprogramDecl::getType() const {
  // All subprogram types contains subprogram type
  return *cast<hir::SubprogramType>(&this->getSymbol().getType());
}

SubprogramDecl::SubprogramDecl(const source::Node &srcNode, const Symbol &sym,
                               FunctionKind kind, std::vector<ParamDecl> params,
                               std::vector<std::unique_ptr<Expr>> exprs)
    : Decl(Node::Kind::SubprogramDecl, srcNode, sym), FuncKind(kind),
      Params(std::move(params)), Expressions(std::move(exprs)) {}

SubprogramDecl::~SubprogramDecl() {}

StringLiteral tmplang::hir::ToString(SubprogramDecl::FunctionKind kind) {
  switch (kind) {
  case SubprogramDecl::proc:
    return "proc";
  case SubprogramDecl::fn:
    return "fn";
  }
  llvm_unreachable("All cases covered");
}
