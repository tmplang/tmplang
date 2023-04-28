#include <llvm/Support/Debug.h>
#include <tmplang/Tree/Source/Expr.h>
#include <tmplang/Tree/Source/RecursiveNodeVisitor.h>
#include <tmplang/Tree/Source/RecursiveTypeVisitor.h>

#include "../TreeFormatPrinter.h"

using namespace tmplang;
using namespace tmplang::source;

namespace {

static constexpr TerminalColor AddressColor = {raw_ostream::YELLOW, false};
static constexpr TerminalColor IdentifierColor = {raw_ostream::GREEN, false};
static constexpr TerminalColor ErrorRecoveryColor = {raw_ostream::RED, false};
static constexpr TerminalColor SourceLocationColor = {raw_ostream::CYAN, true};
static constexpr TerminalColor NodeColor = {raw_ostream::GREEN, true};
static constexpr TerminalColor TypeColor = {raw_ostream::BLUE, true};

class RecursiveSourcePrinter
    : protected TextTreeStructure,
      public RecursiveASTVisitor<RecursiveSourcePrinter>,
      public RecursiveTypeVisitor<RecursiveSourcePrinter> {
public:
  using SrcBase = RecursiveASTVisitor<RecursiveSourcePrinter>;
  using TypeBase = RecursiveTypeVisitor<RecursiveSourcePrinter>;

  RecursiveSourcePrinter(raw_ostream &os, const SourceManager &sm,
                         Node::PrintConfig cfg)
      : TextTreeStructure(os, cfg & Node::PrintConfig::Color), OS(os), SM(sm),
        Cfg(cfg) {}

  bool visitNode(const Node &node) {
    const bool showColors = Cfg & Node::Color;
    {
      ColorScope color(OS, showColors, NodeColor);
      OS << ToString(node.getKind());
    }

    if (Cfg & Node::PrintConfig::Address) {
      PrintPointer(&node, showColors, AddressColor, OS);
    }

    if (Cfg & Node::PrintConfig::SourceLocation) {
      PrintSourceLocation({node.getBeginLoc(), node.getEndLoc()}, showColors,
                          SourceLocationColor, SM, OS);
    }
    OS << ": ";

    SrcBase::visitNode(node);
    return true;
  }

  bool traverseNode(const Node &node) {
    AddChild([&] { SrcBase::traverseNode(node); });
    return true;
  }

  //=--------------------------------------------------------------------------=//
  // End node printing functions
  //=--------------------------------------------------------------------------=//
  bool visitCompilationUnit(const CompilationUnit &compUnit) { return true; }

  bool visitSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    printToken(subprogramDecl.getFuncType());
    OS << ' ';
    printToken(subprogramDecl.getIdentifier());
    if (auto &colon = subprogramDecl.getColon()) {
      OS << ' ';
      printToken(*colon);
    }
    if (auto arrow = subprogramDecl.getArrow()) {
      OS << ' ';
      printToken(*arrow);
    }
    if (auto *retType = subprogramDecl.getReturnType()) {
      OS << ' ';
      traverseType(*retType);
    }
    OS << ' ';
    printToken(subprogramDecl.getLKeyBracket());
    // FIXME: Find a proper location for the right key bracket once we have a
    // body
    OS << ' ';
    printToken(subprogramDecl.getRKeyBracket());
    return true;
  }

  bool visitParamDecl(const ParamDecl &paramDecl) {
    traverseType(paramDecl.getType());
    OS << ' ';
    printToken(paramDecl.getIdentifier());
    if (auto &comma = paramDecl.getComma()) {
      OS << ' ';
      printToken(*comma);
    }
    return true;
  }

  bool visitDataFieldDecl(const DataFieldDecl &dataFieldDecl) {
    printToken(dataFieldDecl.getIdentifier());
    OS << ' ';
    printToken(dataFieldDecl.getColon());
    OS << ' ';
    traverseType(dataFieldDecl.getType());
    if (auto &comma = dataFieldDecl.getComma()) {
      OS << ' ';
      printToken(*comma);
    }
    return true;
  }

  bool visitDataDecl(const DataDecl &dataDecl) {
    printToken(dataDecl.getDataKeyword());
    OS << ' ';
    printToken(dataDecl.getIdentifier());
    OS << ' ';
    printToken(dataDecl.getStartingEq());
    OS << ' ';
    printToken(dataDecl.getEndingSemicolon());

    return true;
  }

  bool visitExprStmt(const ExprStmt &exprStmt) {
    printToken(exprStmt.getSemicolon());
    return true;
  }
  bool visitExprIntegerNumber(const ExprIntegerNumber &exprIntegerNumber) {
    printToken(exprIntegerNumber.getNumber());
    return true;
  }
  bool visitExprTuple(const ExprTuple &exprTuple) {
    printToken(exprTuple.getLParen());
    printToken(exprTuple.getRParen());
    return true;
  }
  bool visitTupleElem(const TupleElem &tupleElem) {
    if (auto &comma = tupleElem.getComma()) {
      printToken(*comma);
    }
    return true;
  }
  bool visitExprRet(const ExprRet &exprRet) {
    printToken(exprRet.getRetTk());
    return true;
  }
  bool visitExprVarRef(const ExprVarRef &exprVarRef) {
    printToken(exprVarRef.getIdentifier());
    return true;
  }
  bool
  visitExprAggregateDataAccess(const ExprAggregateDataAccess &exprDataField) {
    printToken(exprDataField.getDot());
    printToken(exprDataField.getAccessedField());
    return true;
  }

  bool visitExprMatch(const ExprMatch &exprMatch) {
    printToken(exprMatch.getMatch());
    OS << ' ';
    printToken(exprMatch.getLKeyBracket());
    OS << ' ';
    printToken(exprMatch.getRKeyBracket());
    return true;
  }

  bool visitExprMatchCase(const ExprMatchCase &matchCase) {
    printToken(matchCase.getArrow());
    return true;
  }

  bool visitPlaceholderDecl(const PlaceholderDecl &placeholderDecl) {
    printToken(placeholderDecl.getIdentifier());
    return true;
  }

  bool visitDataDestructuration(const DataDestructuration &dataDes) {
    printToken(dataDes.LhsBracket);
    printToken(dataDes.RhsBracket);
    return true;
  }

  bool visitTupleDestructuration(const TupleDestructuration &tupleDes) {
    printToken(tupleDes.getLhsParen());
    printToken(tupleDes.getRhsParen());
    return true;
  }

  bool
  visitDataDestructurationElem(const DataDestructurationElem &dataDesElem) {
    printToken(dataDesElem.getId());
    OS << ' ';
    printToken(dataDesElem.getColon());
    if (auto comma = dataDesElem.getComma()) {
      OS << ' ';
      printToken(*comma);
    }

    return true;
  }

  bool
  visitTupleDestructurationElem(const TupleDestructurationElem &tupleDesElem) {
    if (auto comma = tupleDesElem.getComma()) {
      printToken(*comma);
    }
    return true;
  }

  bool visitUnionDecl(const UnionDecl &unionDecl) {
    printToken(unionDecl.getUnionKeyword());
    OS << ' ';
    printToken(unionDecl.getIdentifier());
    OS << ' ';
    printToken(unionDecl.getStartingEq());
    OS << ' ';
    printToken(unionDecl.getEndingSemicolon());

    return true;
  }

  bool visitUnionAlternativeDecl(const UnionAlternativeDecl &alternativeDecl) {
    printToken(alternativeDecl.getIdentifier());
    OS << ' ';
    printToken(alternativeDecl.getLhsParen());
    OS << ' ';
    printToken(alternativeDecl.getRhsParen());
    OS << ' ';

    return true;
  }

  bool visitUnionAlternativeFieldDecl(const UnionAlternativeFieldDecl &field) {
    printToken(field.getIdentifier());
    OS << ' ';
    printToken(field.getColon());
    OS << ' ';
    traverseType(field.getType());
    if (auto comma = field.getComma()) {
      OS << ' ';
      printToken(*comma);
    }
    return true;
  }

  bool visitUnionDestructuration(const UnionDestructuration &unionDes) {
    printToken(unionDes.getAlternative());
    return true;
  }

  //=--------------------------------------------------------------------------=//
  // End node printing functions
  //=--------------------------------------------------------------------------=//

  //=--------------------------------------------------------------------------=//
  // Begin type printing functions
  //=--------------------------------------------------------------------------=//
  bool visitType(const Type &type) {
    ColorScope color(OS, Cfg & Node::Color, TypeColor);
    TypeBase::visitType(type);
    return true;
  }

  bool visitNamedType(const NamedType &namedType) {
    printToken(namedType.getIdentifier());
    return true;
  }

  bool traverseTupleType(const TupleType &tupleType) {
    printToken(tupleType.getLParentheses());

    ArrayRef<RAIIType> types = tupleType.getTypes();
    ArrayRef<SpecificToken<TK_Comma>> commas = tupleType.getCommas();

    for (unsigned i = 0; i < commas.size(); i++) {
      const RAIIType &type = types[i];
      const SpecificToken<TK_Comma> &comma = commas[i];

      TypeBase::traverseType(*type);
      OS << ' ';
      printToken(comma);
      OS << ' ';
    }

    // Print last type
    if (types.size()) {
      TypeBase::traverseType(*types.back());
    }

    printToken(tupleType.getRParentheses());
    return true;
  }
  //=--------------------------------------------------------------------------=//
  // End type printing functions
  //=--------------------------------------------------------------------------=//

private:
  void printToken(const Token &tk) {
    const bool showColors = Cfg & Node::Color;

    const bool isErrorRecoveryToken = tk.isErrorRecoveryToken();
    if (isErrorRecoveryToken) {
      ColorScope color(OS, showColors, ErrorRecoveryColor);
      OS << "Err![";
    }

    {
      ColorScope color(OS, showColors, IdentifierColor);
      OS << tk.getLexeme();
    }

    if (isErrorRecoveryToken) {
      ColorScope color(OS, showColors, ErrorRecoveryColor);
      OS << ']';
    }
  }

private:
  raw_ostream &OS;
  const SourceManager &SM;
  const Node::PrintConfig Cfg;
};

} // namespace

void tmplang::source::Node::print(raw_ostream &os, const SourceManager &sm,
                                  PrintConfig cfg) const {
  RecursiveSourcePrinter(os, sm, cfg).traverseNode(*this);
}

void tmplang::source::Node::dump(const SourceManager &sm,
                                 PrintConfig cfg) const {
  // FIXME: Set our own debug streams
  if (llvm::dbgs().has_colors()) {
    llvm::dbgs().enable_colors(true);
  }

  print(llvm::dbgs(), sm, cfg);

  llvm::dbgs().enable_colors(false);
}
