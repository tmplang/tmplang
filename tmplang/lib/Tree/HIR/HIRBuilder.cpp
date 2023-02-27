#include <tmplang/Tree/HIR/HIRBuilder.h>

#include <llvm/ADT/StringSet.h>
#include <llvm/Support/Casting.h>
#include <tmplang/Tree/HIR/Exprs.h>
#include <tmplang/Tree/Source/CompilationUnit.h>
#include <tmplang/Tree/Source/Exprs.h>

using namespace tmplang;
using namespace tmplang::hir;

namespace {

class HIRBuilder {
public:
  HIRBuilder(HIRContext &ctx)
      : Ctx(ctx), SymMan(ctx.getSymbolManager()),
        ScopeStack(1, &SymMan.getGlobalScope()) {}

  std::optional<CompilationUnit> build(const source::CompilationUnit &);

private:
  std::unique_ptr<Decl> getTopLevelDecl(const source::Decl &);

  // All declarations
  std::unique_ptr<Decl> get(const source::SubprogramDecl &);
  std::optional<ParamDecl> get(const source::ParamDecl &);
  std::unique_ptr<Decl> get(const source::DataDecl &);
  std::optional<DataFieldDecl> get(const source::DataFieldDecl &);
  std::optional<PlaceholderDecl> get(const source::PlaceholderDecl &,
                                     const Type &matchedExprTy);
  std::unique_ptr<UnionDecl> get(const source::UnionDecl &);
  std::optional<UnionAlternativeDecl> get(const source::UnionAlternativeDecl &);
  std::optional<UnionAlternativeFieldDecl>
  get(const source::UnionAlternativeFieldDecl &);

  // All expressions
  std::unique_ptr<Expr> get(const source::Expr &);
  std::unique_ptr<ExprIntegerNumber> get(const source::ExprIntegerNumber &);
  std::unique_ptr<ExprRet> get(const source::ExprRet &);
  std::unique_ptr<ExprTuple> get(const source::ExprTuple &);
  std::unique_ptr<ExprVarRef> get(const source::ExprVarRef &);
  std::unique_ptr<ExprAggregateDataAccess>
  get(const source::ExprAggregateDataAccess &);

  // MatchExpr related building functions
  std::optional<AggregateDestructurationElem>
  get(const source::AggregateDestructurationElem &, const Type &, unsigned idx);
  std::optional<AggregateDestructuration>
  get(const source::DataDestructuration &, const Type &);
  std::optional<AggregateDestructuration>
  get(const source::TupleDestructuration &, const Type &);
  std::unique_ptr<ExprMatchCaseLhsVal> get(const source::ExprMatchCaseLhsVal &,
                                           const Type &);
  std::unique_ptr<ExprMatchCase> get(const source::ExprMatchCase &,
                                     const Type &matchedExprTy);
  std::unique_ptr<ExprMatch> get(const source::ExprMatch &);

  // All types
  const Type *get(const source::Type &);

  Symbol *fetchSymbolRecursively(SymbolKind kind, llvm::StringRef id) const;
  bool isSymbolInCurrentScope(SymbolKind kind, llvm::StringRef id) const;
  Symbol &addSymbolToCurrentScope(SymbolKind kind, llvm::StringRef id,
                                  const Type &ty,
                                  const SymbolicScope *symScope = nullptr);

  [[maybe_unused]] SymbolicScope &pushSymbolicScope() {
    SymbolicScope &newScope = SymMan.createSymbolicScope();
    return *ScopeStack.emplace_back(&newScope);
  }

  [[maybe_unused]] SymbolicScope &popSymbolScope() {
    return *ScopeStack.pop_back_val();
  }

private:
  HIRContext &Ctx;
  SymbolManager &SymMan;
  /// Stack of scopes we are currently in. SymbolicScopes live in the
  /// SymbolicManager. TODO: Profile to know what size to give initialy
  SmallVector<SymbolicScope *, 6> ScopeStack;
  /// Data Types to its Symbol map
  llvm::DenseMap<const DataType *, const Symbol *> DataTyToSymMap;
};

std::unique_ptr<ExprIntegerNumber>
HIRBuilder::get(const source::ExprIntegerNumber &exprIntNum) {
  // FIXME: For now all numbers are signed 32 bits
  return std::make_unique<hir::ExprIntegerNumber>(
      exprIntNum, hir::BuiltinType::get(Ctx, BuiltinType::K_i32),
      llvm::APInt(32, exprIntNum.getNumber().getNumber(), /*isSigned=*/true));
}

std::unique_ptr<ExprRet> HIRBuilder::get(const source::ExprRet &exprRet) {
  std::unique_ptr<Expr> returnedExpr;
  if (auto *retExpr = exprRet.getReturnedExpr()) {
    returnedExpr = get(*exprRet.getReturnedExpr());
    if (!returnedExpr) {
      // Already reported
      return nullptr;
    }
  }

  return std::make_unique<hir::ExprRet>(
      exprRet, returnedExpr ? returnedExpr->getType() : TupleType::getUnit(Ctx),
      std::move(returnedExpr));
}

std::unique_ptr<ExprTuple> HIRBuilder::get(const source::ExprTuple &exprTuple) {
  SmallVector<std::unique_ptr<hir::Expr>, 4> values;
  transform(exprTuple.getVals(), std::back_inserter(values),
            [&](const source::TupleElem &elem) { return get(elem.getVal()); });

  SmallVector<const Type *> tupleTys;
  tupleTys.reserve(values.size());
  transform(values, std::back_inserter(tupleTys),
            [&](std::unique_ptr<hir::Expr> &expr) { return &expr->getType(); });

  return std::make_unique<hir::ExprTuple>(
      exprTuple, TupleType::get(Ctx, tupleTys), std::move(values));
}

std::unique_ptr<ExprVarRef>
HIRBuilder::get(const source::ExprVarRef &exprVarRef) {
  auto *sym = fetchSymbolRecursively(SymbolKind::ReferenciableFromExprVarRef,
                                     exprVarRef.getName());
  if (!sym) {
    // TODO: Emit error about var not defined
    return nullptr;
  }

  return std::make_unique<ExprVarRef>(exprVarRef, *sym);
}

std::unique_ptr<ExprMatchCaseLhsVal>
HIRBuilder::get(const source::ExprMatchCaseLhsVal &lhsVal, const Type &currTy) {
  auto visitors = source::visitors{
      [&](const std::unique_ptr<source::Expr> &expr) {
        auto evaluatedExpr = get(*expr);
        return evaluatedExpr ? std::make_unique<ExprMatchCaseLhsVal>(
                                   std::move(evaluatedExpr))
                             : nullptr;
      },
      [](const source::VoidPlaceholder &arg) {
        return std::make_unique<ExprMatchCaseLhsVal>(VoidPlaceholder());
      },
      [&](const source::PlaceholderDecl &arg) {
        auto placeHolder = get(arg, currTy);
        return placeHolder ? std::make_unique<ExprMatchCaseLhsVal>(
                                 std::move(*placeHolder))
                           : nullptr;
      },
      [&](const source::TupleDestructuration &arg) {
        auto des = get(arg, currTy);
        return des ? std::make_unique<ExprMatchCaseLhsVal>(std::move(*des))
                   : nullptr;
      },
      [&](const source::DataDestructuration &arg) {
        auto des = get(arg, currTy);
        return des ? std::make_unique<ExprMatchCaseLhsVal>(std::move(*des))
                   : nullptr;
      },
      [](const auto &arg) -> std::unique_ptr<ExprMatchCaseLhsVal> {
        llvm_unreachable("All cases covered");
      }};

  return std::visit(visitors, lhsVal);
}

std::optional<AggregateDestructurationElem>
HIRBuilder::get(const source::AggregateDestructurationElem &srcNode,
                const Type &currTy, unsigned idx) {
  auto value = get(srcNode.getValue(), currTy);
  return value ? AggregateDestructurationElem(srcNode, currTy, idx,
                                              std::move(value))
               : std::optional<AggregateDestructurationElem>{};
}

std::optional<AggregateDestructuration>
HIRBuilder::get(const source::DataDestructuration &srcNode,
                const Type &currentTy) {
  std::vector<AggregateDestructurationElem> destructuratedElems;
  destructuratedElems.reserve(srcNode.DataElems.size());

  llvm::StringSet<> alreadySeenFields;
  for (const source::DataDestructurationElem &elem : srcNode.DataElems) {
    auto *dataTy = dyn_cast<DataType>(&currentTy);
    if (!dataTy) {
      // TODO: Emit error about destructuring a non data type
      return std::nullopt;
    }

    assert(DataTyToSymMap.count(dataTy));

    const SymbolicScope *dataTyScope =
        DataTyToSymMap[dataTy]->getCreatedSymScope();

    // Find if the field exists in the type
    auto *fieldSymIt = llvm::find_if(
        dataTyScope->getSymbols(), [&currElem = elem](const Symbol &sym) {
          return sym.getId() == currElem.getId().getLexeme();
        });

    if (fieldSymIt == dataTyScope->getSymbols().end()) {
      // TODO: Emit error about accessing an element which does not exists
      return std::nullopt;
    }

    // Field aready accessed, this is an error
    if (!alreadySeenFields.insert(fieldSymIt->get().getId()).second) {
      // TODO: Emit error about field aready accessed
      return std::nullopt;
    }

    // Pass the type of the retrieve field
    auto val =
        get(elem, fieldSymIt->get().getType(),
            std::distance(dataTyScope->getSymbols().begin(), fieldSymIt));
    if (!val) {
      // Error already reported
      return std::nullopt;
    }

    destructuratedElems.push_back(std::move(*val));
  }

  return AggregateDestructuration(srcNode, currentTy,
                                  std::move(destructuratedElems));
}

std::optional<AggregateDestructuration>
HIRBuilder::get(const source::TupleDestructuration &srcNode,
                const Type &currentTy) {
  std::vector<AggregateDestructurationElem> destructuratedElems;
  destructuratedElems.reserve(srcNode.getTupleElems().size());

  for (const auto &idxAndElem : llvm::enumerate(srcNode.getTupleElems())) {
    auto [idx, elem] = idxAndElem;

    auto *tupleTy = dyn_cast<TupleType>(&currentTy);
    if (!tupleTy) {
      // TODO: Emit error about destructuring a non tuple type
      return std::nullopt;
    }

    if (idx >= tupleTy->getTypes().size()) {
      // TODO: emir error about accesing the elements of the tuple
      return std::nullopt;
    }

    // Pass the type of the retrieve elemt of the tuple
    auto val = get(elem, *tupleTy->getTypes()[idx], idx);
    if (!val) {
      // Error already reported
      return std::nullopt;
    }

    destructuratedElems.push_back(std::move(*val));
  }

  return AggregateDestructuration(srcNode, currentTy,
                                  std::move(destructuratedElems));
}

std::unique_ptr<ExprMatchCase>
HIRBuilder::get(const source::ExprMatchCase &matchCase,
                const Type &matchedExprTy) {
  pushSymbolicScope();

  std::optional<ExprMatchCase::LhsValue> matchingElem = std::visit(
      source::visitors{
          [&](const source::Otherwise &arg)
              -> std::optional<ExprMatchCase::LhsValue> {
            return ExprMatchCase::LhsValue{Otherwise()};
          },
          [&](const std::unique_ptr<source::ExprMatchCaseLhsVal> &arg)
              -> std::optional<ExprMatchCase::LhsValue> {
            auto lhsValue = get(*arg, matchedExprTy);
            return lhsValue ? ExprMatchCase::LhsValue(std::move(*lhsValue))
                            : std::optional<ExprMatchCase::LhsValue>{};
          }},
      matchCase.getLhs());

  if (!matchingElem) {
    // Errors already reported
    return nullptr;
  }

  auto expr = get(*matchCase.getRhs());
  if (!expr) {
    // Error already reported
    return nullptr;
  }

  popSymbolScope();

  return std::make_unique<ExprMatchCase>(matchCase, std::move(*matchingElem),
                                         std::move(expr));
}

std::unique_ptr<ExprMatch> HIRBuilder::get(const source::ExprMatch &exprMatch) {
  auto matchedExpr = get(exprMatch.getMatchedExpr());
  if (!matchedExpr) {
    return nullptr;
  }

  assert(!exprMatch.getCases().empty() && "Guaranteed by grammar");

  SmallVector<std::unique_ptr<ExprMatchCase>> cases;
  cases.reserve(exprMatch.getCases().size());

  for (const source::ExprMatchCase &matchCase : exprMatch.getCases()) {
    auto hirCase = get(matchCase, matchedExpr->getType());
    if (!hirCase) {
      return nullptr;
    }
    cases.push_back(std::move(hirCase));
  }

  return std::make_unique<ExprMatch>(exprMatch, cases.front()->getType(),
                                     std::move(matchedExpr), std::move(cases));
}

static std::unique_ptr<ExprAggregateDataAccess>
GetTupleAccess(const source::ExprAggregateDataAccess &exprAggregateDataAccess,
               const TupleType &tupleTy, std::unique_ptr<Expr> baseExpr) {
  if (exprAggregateDataAccess.getAccessedField().isNot(TK_IntegerNumber)) {
    // TODO: Emit error about tuples must be accessed though integer numbers
    return nullptr;
  }

  int32_t idx = exprAggregateDataAccess.getNumber();
  if (static_cast<uint32_t>(idx) >= tupleTy.getTypes().size()) {
    // TODO: Emit error about bad index
    return nullptr;
  }

  return std::make_unique<hir::ExprAggregateDataAccess>(
      exprAggregateDataAccess,
      ExprAggregateDataAccess::FromExprToAggregateDataAccOrVarRefOrTuple(
          std::move(baseExpr)),
      *tupleTy.getTypes()[idx], idx);
}

static std::unique_ptr<ExprAggregateDataAccess> GetDataAccess(
    const source::ExprAggregateDataAccess &exprAggregateDataAccess,
    const DataType &dataTy, std::unique_ptr<Expr> baseExpr,
    llvm::DenseMap<const DataType *, const Symbol *> &dataTyToSymMap) {
  if (exprAggregateDataAccess.getAccessedField().isNot(TK_Identifier)) {
    // TODO: Emit error about data types must be accessed though identifiers
    return nullptr;
  }

  // Find Symbol of DataType
  assert(dataTyToSymMap.count(&dataTy) && "The type must be in the map");
  auto &dataTySym = *dataTyToSymMap[&dataTy];

  const SymbolicScope *createdSymScopeByBaseSym =
      dataTySym.getCreatedSymScope();
  assert(createdSymScopeByBaseSym &&
         "All DataType symbols create a symbolic scope for its fields");

  std::optional<unsigned> idxOfField;
  const Symbol *fieldSym = nullptr;
  for (auto idxAndFieldSym :
       llvm::enumerate(createdSymScopeByBaseSym->getSymbols())) {
    auto &[idx, sym] = idxAndFieldSym;
    if (sym.get().getId() == exprAggregateDataAccess.getFieldName()) {
      idxOfField = idx;
      fieldSym = &sym.get();
      break;
    }
  }
  if (!idxOfField || !fieldSym) {
    // TODO: Emit error about no named field for type X
    return nullptr;
  }

  return std::make_unique<hir::ExprAggregateDataAccess>(
      exprAggregateDataAccess,
      ExprAggregateDataAccess::FromExprToAggregateDataAccOrVarRefOrTuple(
          std::move(baseExpr)),
      *fieldSym, *idxOfField);
}

std::unique_ptr<ExprAggregateDataAccess> HIRBuilder::get(
    const source::ExprAggregateDataAccess &exprAggregateDataAccess) {
  auto base = get(exprAggregateDataAccess.getBase());
  if (!base) {
    // Error already reported
    return nullptr;
  }

  const Type &baseTy = base->getType();
  auto *tupleTy = dyn_cast<TupleType>(&baseTy);
  auto *dataTy = dyn_cast<DataType>(&baseTy);
  if (!tupleTy && !dataTy) {
    // Return error about trying to access neither tuple nor data type
    return nullptr;
  }

  return tupleTy ? GetTupleAccess(exprAggregateDataAccess, *tupleTy,
                                  std::move(base))
                 : GetDataAccess(exprAggregateDataAccess, *dataTy,
                                 std::move(base), DataTyToSymMap);
}

std::unique_ptr<Expr> HIRBuilder::get(const source::Expr &expr) {
  switch (expr.getKind()) {
  case source::Node::Kind::ExprTuple:
    return get(*cast<source::ExprTuple>(&expr));
  case source::Node::Kind::ExprIntegerNumber:
    return get(*cast<source::ExprIntegerNumber>(&expr));
  case source::Node::Kind::ExprRet:
    return get(*cast<source::ExprRet>(&expr));
  case source::Node::Kind::ExprVarRef:
    return get(*cast<source::ExprVarRef>(&expr));
  case source::Node::Kind::ExprAggregateDataAccess:
    return get(*cast<source::ExprAggregateDataAccess>(&expr));
  case source::Node::Kind::ExprMatch:
    return get(*cast<source::ExprMatch>(&expr));
  case source::Node::Kind::ExprMatchCase:
  case source::Node::Kind::CompilationUnit:
  case source::Node::Kind::SubprogramDecl:
  case source::Node::Kind::DataFieldDecl:
  case source::Node::Kind::DataDecl:
  case source::Node::Kind::ParamDecl:
  case source::Node::Kind::ExprStmt:
  case source::Node::Kind::TupleElem:
  case source::Node::Kind::PlaceholderDecl:
  case source::Node::Kind::VoidPlaceholder:
  case source::Node::Kind::Otherwise:
  case source::Node::Kind::TupleDestructuration:
  case source::Node::Kind::TupleDestructurationElem:
  case source::Node::Kind::DataDestructuration:
  case source::Node::Kind::DataDestructurationElem:
  case source::Node::Kind::UnionDecl:
  case source::Node::Kind::UnionAlternativeDecl:
  case source::Node::Kind::UnionAlternativeFieldDecl:
    break;
  }
  llvm_unreachable("This should not be reachable");
}

const hir::Type *HIRBuilder::get(const source::Type &type) {
  switch (type.getKind()) {
  case source::Type::NamedType: {
    auto *sym =
        fetchSymbolRecursively(SymbolKind::ReferenciableFromType,
                               llvm::cast<source::NamedType>(&type)->getName());
    if (!sym) {
      // TODO: Emit error (type undefined)
      return nullptr;
    }

    return &sym->getType();
  }
  case source::Type::TupleType:
    const auto &tupleType = *cast<source::TupleType>(&type);
    SmallVector<const hir::Type *> types;
    transform(tupleType.getTypes(), std::back_inserter(types),
              [&](const source::RAIIType &type) { return get(*type); });

    if (any_of(types, [](const Type *type) { return type == nullptr; })) {
      return nullptr;
    }
    return &TupleType::get(Ctx, types);
  }
  llvm_unreachable("All cases covered");
}

Symbol *HIRBuilder::fetchSymbolRecursively(SymbolKind kind,
                                           llvm::StringRef id) const {
  for (SymbolicScope *symScope : llvm::reverse(ScopeStack)) {
    auto *sym = llvm::find_if(symScope->getSymbols(), [=](const Symbol &sym) {
      return sym.getKind() == kind && sym.getId() == id;
    });

    if (sym != symScope->getSymbols().end()) {
      return &sym->get();
    }
  }

  return nullptr;
}

bool HIRBuilder::isSymbolInCurrentScope(SymbolKind kind,
                                        llvm::StringRef id) const {
  return ScopeStack.back()->containsSymbol(kind, id);
}

Symbol &HIRBuilder::addSymbolToCurrentScope(SymbolKind kind, llvm::StringRef id,
                                            const Type &ty,
                                            const SymbolicScope *symScope) {
  assert(!isSymbolInCurrentScope(kind, id));

  auto &sym = SymMan.createSymbol(kind, id, ty, symScope);
  ScopeStack.back()->addSymbol(sym);
  return sym;
}

std::optional<ParamDecl>
HIRBuilder::get(const source::ParamDecl &srcParamDecl) {
  const Type *type = get(srcParamDecl.getType());
  if (!type) {
    return nullopt;
  }

  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
                             srcParamDecl.getName())) {
    // FIXME: Report error
    return nullopt;
  }
  Symbol &sym = addSymbolToCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
                                        srcParamDecl.getName(), *type);

  return ParamDecl(srcParamDecl, sym);
}

static SubprogramDecl::FunctionKind
GetFunctionKind(SpecificToken<TK_FnType, TK_ProcType> tk) {
  if (tk.is(TK_ProcType)) {
    return SubprogramDecl::proc;
  }
  return SubprogramDecl::fn;
}

std::unique_ptr<Decl> HIRBuilder::getTopLevelDecl(const source::Decl &decl) {
  switch (decl.getKind()) {
  case source::Node::Kind::SubprogramDecl:
    return get(*cast<source::SubprogramDecl>(&decl));
  case source::Node::Kind::DataDecl:
    return get(*cast<source::DataDecl>(&decl));
  case source::Node::Kind::UnionDecl:
    return get(*cast<source::UnionDecl>(&decl));
  case source::Node::Kind::CompilationUnit:
  case source::Node::Kind::DataFieldDecl:
  case source::Node::Kind::ParamDecl:
  case source::Node::Kind::ExprStmt:
  case source::Node::Kind::ExprIntegerNumber:
  case source::Node::Kind::ExprRet:
  case source::Node::Kind::ExprAggregateDataAccess:
  case source::Node::Kind::ExprTuple:
  case source::Node::Kind::ExprVarRef:
  case source::Node::Kind::TupleElem:
  case source::Node::Kind::ExprMatch:
  case source::Node::Kind::ExprMatchCase:
  case source::Node::Kind::PlaceholderDecl:
  case source::Node::Kind::VoidPlaceholder:
  case source::Node::Kind::Otherwise:
  case source::Node::Kind::TupleDestructuration:
  case source::Node::Kind::TupleDestructurationElem:
  case source::Node::Kind::DataDestructuration:
  case source::Node::Kind::DataDestructurationElem:
  case source::Node::Kind::UnionAlternativeDecl:
  case source::Node::Kind::UnionAlternativeFieldDecl:
    // All these nodes cannot be top level decls
    break;
  }
  llvm_unreachable("These case are not possible");
}

std::unique_ptr<Decl> HIRBuilder::get(const source::SubprogramDecl &srcFunc) {
  const Type *hirReturnType = nullptr;
  if (const source::Type *srcType = srcFunc.getReturnType()) {
    hirReturnType = get(*srcType);
  } else {
    hirReturnType = &TupleType::getUnit(Ctx);
  }
  if (!hirReturnType) {
    return nullptr;
  }

  // Add new scope for params and body
  pushSymbolicScope();

  std::vector<ParamDecl> paramList;
  for (const source::ParamDecl &param : srcFunc.getParams()) {
    auto hirParamDecl = get(param);
    if (!hirParamDecl) {
      return nullptr;
    }
    paramList.push_back(std::move(*hirParamDecl));
  }

  std::vector<std::unique_ptr<Expr>> bodyExprs;
  for (const source::ExprStmt &expr : srcFunc.getBlock().Exprs) {
    auto *srcExpr = expr.getExpr();
    if (!srcExpr) {
      continue;
    }

    std::unique_ptr<hir::Expr> hirExpr = get(*srcExpr);
    if (!hirExpr) {
      return nullptr;
    }

    bodyExprs.push_back(std::move(hirExpr));
  }

  // Pop params and body scope, so we are back previous scope for next
  // declarations
  popSymbolScope();

  SmallVector<const Type *> paramTys;
  transform(paramList, std::back_inserter(paramTys),
            [](const ParamDecl &paramDecl) { return &paramDecl.getType(); });
  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromExprCall,
                             srcFunc.getName())) {
    // FIXME: Report error
    return nullptr;
  }
  Symbol &funcSym = addSymbolToCurrentScope(
      SymbolKind::ReferenciableFromExprCall, srcFunc.getName(),
      SubprogramType::get(Ctx, *hirReturnType, paramTys));

  return std::make_unique<SubprogramDecl>(
      srcFunc, funcSym, GetFunctionKind(srcFunc.getFuncType()),
      std::move(paramList), std::move(bodyExprs));
}

std::unique_ptr<Decl> HIRBuilder::get(const source::DataDecl &dataDecl) {
  // Add new scope for data fields
  pushSymbolicScope();

  std::vector<DataFieldDecl> fields;
  for (const source::DataFieldDecl &dataFieldDecl : dataDecl.getFields()) {
    auto hirFieldDecl = get(dataFieldDecl);
    if (!hirFieldDecl) {
      return nullptr;
    }
    fields.emplace_back(std::move(*hirFieldDecl));
  }

  SmallVector<const Type *> fieldTys;
  fieldTys.reserve(fields.size());
  transform(fields, std::back_inserter(fieldTys),
            [](const DataFieldDecl &field) { return &field.getType(); });

  SymbolicScope &createdScope = popSymbolScope();

  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromType,
                             dataDecl.getName())) {
    // FIXME: Report error
    return nullptr;
  }

  // FIXME: All this type, symbol and decl creation should be made at the same
  // time, since each of them needs of the other. Create a unique point of
  // creation for all three.

  // Get data type
  auto &dataTy = DataType::get(Ctx, dataDecl.getName(), fieldTys);

  // Create symbol for this data type decl
  Symbol &dataDeclSym =
      addSymbolToCurrentScope(SymbolKind::ReferenciableFromType,
                              dataDecl.getName(), dataTy, &createdScope);

  // Link the type with its associated symbol
  DataTyToSymMap[&dataTy] = &dataDeclSym;

  return std::make_unique<DataDecl>(dataDecl, dataDeclSym, std::move(fields));
}

std::optional<DataFieldDecl>
HIRBuilder::get(const source::DataFieldDecl &dataFieldDecl) {
  auto *ty = get(dataFieldDecl.getType());
  if (!ty) {
    return nullopt;
  }

  // FIXME: In reality these should not be stored since they are no Types
  // and do not provide much value. We are resuing the simbolic scopes
  // to block redefinitions of the same data file decl
  if (isSymbolInCurrentScope(SymbolKind::UnreferenciableDataFieldDecl,
                             dataFieldDecl.getName())) {
    // FIXME: Report error
    return nullopt;
  }
  Symbol &dataFieldSym = addSymbolToCurrentScope(
      SymbolKind::UnreferenciableDataFieldDecl, dataFieldDecl.getName(), *ty);

  return DataFieldDecl(dataFieldDecl, dataFieldSym);
}

std::optional<PlaceholderDecl>
HIRBuilder::get(const source::PlaceholderDecl &placeholderDecl,
                const Type &placeholderTy) {
  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
                             placeholderDecl.getName())) {
    // TODO: Report message about placeholder already defined
    return std::nullopt;
  }

  auto &sym = addSymbolToCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
                                      placeholderDecl.getName(), placeholderTy);
  return PlaceholderDecl(placeholderDecl, sym);
}

std::unique_ptr<UnionDecl> HIRBuilder::get(const source::UnionDecl &enumDecl) {
  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromType,
                             enumDecl.getName())) {
    // TODO: Report message about type already defined
    return nullptr;
  }

  auto &unionSymScope = pushSymbolicScope();

  std::vector<UnionAlternativeDecl> alternatives;
  for (auto &alternative : enumDecl.getAlternatives()) {
    auto optAlternative = get(alternative);
    if (!optAlternative) {
      // Error already reported
      return nullptr;
    }
    alternatives.push_back(std::move(*optAlternative));
  }

  popSymbolScope();

  SmallVector<const Type *> alternativeTys;
  alternativeTys.reserve(alternatives.size());
  transform(alternatives, std::back_inserter(alternativeTys),
            [](const UnionAlternativeDecl &field) { return &field.getType(); });

  auto &unionTy = UnionType::get(Ctx, enumDecl.getName(), alternativeTys);

  auto &sym =
      addSymbolToCurrentScope(SymbolKind::ReferenciableFromType,
                              enumDecl.getName(), unionTy, &unionSymScope);

  return std::make_unique<UnionDecl>(enumDecl, sym, std::move(alternatives));
}

std::optional<UnionAlternativeDecl>
HIRBuilder::get(const source::UnionAlternativeDecl &alternative) {
  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromType,
                             alternative.getName())) {
    // TODO: Report message about alternative already defined
    return nullopt;
  }

  pushSymbolicScope();

  std::vector<UnionAlternativeFieldDecl> fields;
  for (auto &field : alternative.getFields()) {
    auto optField = get(field);
    if (!optField) {
      // Error already reported
      return nullopt;
    }
    fields.push_back(std::move(*optField));
  }

  auto &alternativeScope = popSymbolScope();

  SmallVector<const Type *> fieldTys;
  fieldTys.reserve(fields.size());
  transform(
      fields, std::back_inserter(fieldTys),
      [](const UnionAlternativeFieldDecl &field) { return &field.getType(); });

  // Alternative types are struct types
  auto &alternativeTy = DataType::get(Ctx, alternative.getName(), fieldTys);

  auto &sym = addSymbolToCurrentScope(SymbolKind::ReferenciableFromType,
                                      alternative.getName(), alternativeTy,
                                      &alternativeScope);

  // Link the type with its associated symbol
  DataTyToSymMap[&alternativeTy] = &sym;

  return std::make_optional<UnionAlternativeDecl>(alternative, sym,
                                                  std::move(fields));
}

std::optional<UnionAlternativeFieldDecl>
HIRBuilder::get(const source::UnionAlternativeFieldDecl &alternativeFieldDecl) {
  if (isSymbolInCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
                             alternativeFieldDecl.getName())) {
    // TODO: Report message about field already defined
    return nullopt;
  }

  auto *fieldTy = get(alternativeFieldDecl.getType());
  if (!fieldTy) {
    // Already reported
    return nullopt;
  }

  auto &sym = addSymbolToCurrentScope(SymbolKind::ReferenciableFromExprVarRef,
                                      alternativeFieldDecl.getName(), *fieldTy);

  return std::make_optional<UnionAlternativeFieldDecl>(alternativeFieldDecl,
                                                       sym);
}

static void AddNamedBuiltinTypes(SymbolManager &sm, HIRContext &ctx) {
  SymbolicScope &symScope = sm.getGlobalScope();

  constexpr BuiltinType::Kind builtinTypes[] = {BuiltinType::Kind::K_i32};
  for (BuiltinType::Kind kind : builtinTypes) {
    auto &sym = sm.createSymbol(SymbolKind::ReferenciableFromType,
                                ToString(kind), BuiltinType::get(ctx, kind),
                                /*createdSymScope=*/nullptr);
    symScope.addSymbol(sym);
  }
}

std::optional<CompilationUnit>
HIRBuilder::build(const source::CompilationUnit &compUnit) {
  CompilationUnit result(compUnit);

  AddNamedBuiltinTypes(SymMan, Ctx);

  for (const std::unique_ptr<tmplang::source::Decl> &srcDecl :
       compUnit.getTopLevelDecls()) {

    auto decl = getTopLevelDecl(*srcDecl);
    if (!decl) {
      return nullopt;
    }

    result.addDecl(std::move(decl));
  }

  return result;
}

} // namespace

std::optional<CompilationUnit>
tmplang::hir::buildHIR(const source::CompilationUnit &compUnit,
                       HIRContext &ctx) {
  return HIRBuilder(ctx).build(compUnit);
}
