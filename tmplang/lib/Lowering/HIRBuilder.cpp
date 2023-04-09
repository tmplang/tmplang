#include "tmplang/Lexer/Token.h"
#include "tmplang/Lowering/Dialect/HIR/Types.h"
#include "llvm/Support/ErrorHandling.h"
#include <tmplang/Lowering/HIRBuilder.h>

#include <llvm/Support/Debug.h>

#include <tmplang/Lowering/Dialect/HIR/Ops.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/Source/CompilationUnit.h>
#include <tmplang/Tree/Source/Decls.h>
#include <tmplang/Tree/Source/Exprs.h>

using namespace tmplang;

namespace {

static tmplang::SymbolAttr GetSymbolAttr(mlir::OpBuilder &b, StringRef sym,
                                         SymbolKind kind, mlir::Type ty) {
  return tmplang::SymbolAttr::get(
      b.getContext(), mlir::SymbolRefAttr::get(b.getContext(), sym),
      SymbolKindAttr::get(b.getContext(), kind), mlir::TypeAttr::get(ty));
}

static void AddNamedBuiltinTypes(mlir::OpBuilder &builder) {
  builder.create<BuiltinTypeOp>(builder.getUnknownLoc(),
                                GetSymbolAttr(builder, "i32",
                                              SymbolKind::BuiltinType,
                                              builder.getI32Type()));
}

class HIRBuilder {
public:
  HIRBuilder(mlir::MLIRContext &ctx, const SourceManager &sm)
      : Ctx(ctx), SM(sm),
        FilePath(mlir::StringAttr::get(&ctx, sm.getFilePath())), B(&Ctx) {}

  TranslationUnitOp build(const source::CompilationUnit &compUnit) {
    auto tuOp =
        B.create<TranslationUnitOp>(B.getUnknownLoc(), SM.getFileName());
    auto &uniqueTUBlock = tuOp.getBody().emplaceBlock();
    B.setInsertionPointToEnd(&uniqueTUBlock);

    AddNamedBuiltinTypes(B);

    for (const std::unique_ptr<tmplang::source::Decl> &srcDecl :
         compUnit.getTopLevelDecls()) {
      B.setInsertionPointToEnd(&uniqueTUBlock);
      get(*srcDecl);
    }

    return tuOp;
  }

private:
  void build(const source::SubprogramDecl &subprog) {
    SmallVector<mlir::Type, 4> argTys;
    transform(
        subprog.getParams(), std::back_inserter(argTys),
        [&](const source::ParamDecl &param) { return get(param.getType()); });

    mlir::Type retTy = subprog.getReturnType()
                           ? get(*subprog.getReturnType())
                           : B.getTupleType(mlir::TypeRange{});
    assert(retTy && "This cannot fail");

    auto subprogramOp = B.create<SubprogramOp>(
        getLocation(subprog.getFuncType()),
        GetSymbolAttr(B, subprog.getName(), SymbolKind::Subprogram,
                      mlir::FunctionType::get(&Ctx, argTys, retTy)));
    B.setInsertionPointToEnd(&subprogramOp.getBody().emplaceBlock());

    for (auto &param : subprog.getParams()) {
      build(param);
    }

    for (auto &expr : subprog.getBlock().Exprs) {
      get(*expr.getExpr());
    }
  }

  void build(const source::ParamDecl &paramDecl) {
    B.create<SubprogramParamOp>(getLocation(paramDecl.getIdentifier()),
                                GetSymbolAttr(B, paramDecl.getName(),
                                              SymbolKind::VarDecl,
                                              get(paramDecl.getType())));
  }

  mlir::Value get(const source::ExprVarRef &exprVarRef) {
    return B.create<VarRefOp>(
        getLocation(exprVarRef.getIdentifier()),
        UnresolvedSymbolType::get(B.getContext()),
        mlir::SymbolRefAttr::get(B.getContext(), exprVarRef.getName()));
  }

  mlir::Value get(const source::ExprAggregateDataAccess &aggregateAccess) {
    mlir::Value base = get(aggregateAccess.getBase());

    if (aggregateAccess.getAccessedField().is(TK_Identifier)) {
      return B.create<DataAccessOp>(
          getLocation(aggregateAccess.getDot()),
          UnresolvedSymbolType::get(B.getContext()), base,
          mlir::SymbolRefAttr::get(B.getContext(),
                                   aggregateAccess.getFieldName()));
    }

    if (aggregateAccess.getAccessedField().is(TK_IntegerNumber)) {
      return B.create<TupleAccessOp>(
          getLocation(aggregateAccess.getDot()),
          UnresolvedSymbolType::get(B.getContext()), base,
          B.getIndexAttr(aggregateAccess.getNumber()));
    }

    llvm_unreachable("");
  }

  void build(const source::ExprRet &exprRet) {
    auto *returnedExpr = exprRet.getReturnedExpr();
    returnedExpr ? B.create<ReturnOp>(getLocation(exprRet.getRetTk()),
                                      get(*returnedExpr))
                 : B.create<ReturnOp>(getLocation(exprRet.getRetTk()));
  }

  mlir::Value get(const source::Node &expr) {
    switch (expr.getKind()) {
    case source::Node::Kind::ExprStmt:
      return get(*cast<source::ExprStmt>(&expr)->getExpr());
    case source::Node::Kind::ExprAggregateDataAccess:
      return get(*cast<source::ExprAggregateDataAccess>(&expr));
    case source::Node::Kind::ExprVarRef:
      return get(*cast<source::ExprVarRef>(&expr));
    case source::Node::Kind::ExprRet:
      build(*cast<source::ExprRet>(&expr));
      return {};
    case source::Node::Kind::SubprogramDecl:
      build(*cast<source::SubprogramDecl>(&expr));
      return {};
    case source::Node::Kind::DataDecl:
      build(*cast<source::DataDecl>(&expr));
      return {};
    case source::Node::Kind::DataFieldDecl:
      build(*cast<source::DataFieldDecl>(&expr));
      return {};
    case source::Node::Kind::CompilationUnit:
    case source::Node::Kind::ParamDecl:
    case source::Node::Kind::ExprIntegerNumber:
    case source::Node::Kind::ExprTuple:
    case source::Node::Kind::TupleElem:
    case source::Node::Kind::ExprMatch:
    case source::Node::Kind::PlaceholderDecl:
    case source::Node::Kind::ExprMatchCase:
    case source::Node::Kind::TupleDestructuration:
    case source::Node::Kind::TupleDestructurationElem:
    case source::Node::Kind::DataDestructuration:
    case source::Node::Kind::DataDestructurationElem:
    case source::Node::Kind::VoidPlaceholder:
    case source::Node::Kind::Otherwise:
      return {};
    }
  }

  void build(const source::DataDecl &dataDecl) {
    mlir::OpBuilder::InsertionGuard guard(B);

    auto dataDeclOp = B.create<DataDeclOp>(
        getLocation(dataDecl.getDataKeyword()),
        GetSymbolAttr(B, dataDecl.getName(), SymbolKind::DataType,
                      DataType::get(B.getContext(), dataDecl.getName())));
    B.setInsertionPointToEnd(&dataDeclOp.getBody().emplaceBlock());

    for (auto &dataFieldDecl : dataDecl.getFields()) {
      build(dataFieldDecl);
    }
  }

  void build(const source::DataFieldDecl &dataFieldDecl) {
    B.create<DataFieldDeclOp>(getLocation(dataFieldDecl.getIdentifier()),
                              GetSymbolAttr(B, dataFieldDecl.getName(),
                                            SymbolKind::DataField,
                                            get(dataFieldDecl.getType())));
  }

  mlir::Type get(const source::Type &ty) {
    switch (ty.getKind()) {
    case source::Type::NamedType:
      return get(*cast<source::NamedType>(&ty));
    case source::Type::TupleType:
      return get(*cast<source::TupleType>(&ty));
    }
    llvm_unreachable("All cases covered");
  }

  const mlir::Type get(const source::NamedType &ty) {
    return UnresolvedType::get(B.getContext(), ty.getName());
  }

  const mlir::Type get(const source::TupleType &ty) {
    SmallVector<mlir::Type, 4> tupleTys;

    transform(ty.getTypes(), std::back_inserter(tupleTys),
              [&](const source::RAIIType &tupleTy) { return get(*tupleTy); });

    return B.getTupleType(tupleTys);
  }

  mlir::FileLineColLoc getLocation(tmplang::Token tk) {
    const LineAndColumn lineAndCol = SM.getLineAndColumn(tk.getSpan().Start);
    return mlir::FileLineColLoc::get(FilePath, lineAndCol.Line,
                                     lineAndCol.Column);
  }

  mlir::SymbolRefAttr getSymbolRefAttr(const StringRef &name) {
    return mlir::SymbolRefAttr::get(&Ctx, name);
  }

  SymbolKindAttr getSymbolKindAttr(SymbolKind kind) {
    return SymbolKindAttr::get(&Ctx, kind);
  }

private:
  mlir::MLIRContext &Ctx;
  const SourceManager &SM;
  const mlir::StringAttr FilePath;
  mlir::OpBuilder B;
};

} // namespace

TranslationUnitOp tmplang::LowerToHIR(const source::CompilationUnit &compUnit,
                                      mlir::MLIRContext &ctx,
                                      const SourceManager &sm) {
  return HIRBuilder(ctx, sm).build(compUnit);
}
