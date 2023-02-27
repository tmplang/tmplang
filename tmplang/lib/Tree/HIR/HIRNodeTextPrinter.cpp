#include <llvm/Support/Debug.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/RecursiveNodeVisitor.h>
#include <tmplang/Tree/HIR/RecursiveTypeVisitor.h>

#include "../TreeFormatPrinter.h"

using namespace tmplang;
using namespace tmplang::hir;

namespace {

static constexpr TerminalColor AddressColor = {raw_ostream::YELLOW, false};
static constexpr TerminalColor IdentifierColor = {raw_ostream::GREEN, false};
static constexpr TerminalColor AttrColor = {raw_ostream::RED, false};
static constexpr TerminalColor LiteralColor = {raw_ostream::RED, false};
static constexpr TerminalColor SourceLocationColor = {raw_ostream::CYAN, true};
static constexpr TerminalColor NodeColor = {raw_ostream::GREEN, true};
static constexpr TerminalColor TypeColor = {raw_ostream::BLUE, true};

class RecursiveHIRTypePrinterBase
    : public RecursiveTypeVisitor<RecursiveHIRTypePrinterBase> {
  using TypeBase = RecursiveTypeVisitor<RecursiveHIRTypePrinterBase>;

public:
  RecursiveHIRTypePrinterBase(raw_ostream &os,
                              Node::PrintConfig cfg = Node::PrintConfig::None)
      : OS(os), Cfg(cfg) {}
  //=--------------------------------------------------------------------------=//
  // Begin type printing functions
  //=--------------------------------------------------------------------------=//
  bool visitType(const Type &type) {
    ColorScope color(OS, Cfg & Node::PrintConfig::Color, TypeColor);
    TypeBase::visitType(type);
    return true;
  }

  bool visitBuiltinType(const BuiltinType &builtinType) {
    OS << ToString(builtinType.getBuiltinKind());
    return true;
  }

  bool traverseTupleType(const TupleType &tupleType) {
    OS << "(";
    llvm::interleaveComma(tupleType.getTypes(), OS, [&](const Type *type) {
      TypeBase::traverseType(*type);
    });
    OS << ")";
    return true;
  }

  void printAggregatedType(StringRef name) {
    ColorScope color(OS, Cfg & Node::PrintConfig::Color, TypeColor);
    OS << name << "{...}";
  }

  bool traverseDataType(const DataType &dataType) {
    printAggregatedType(dataType.getName());
    return true;
  }

  bool traverseUnionType(const UnionType &unionType) {
    printAggregatedType(unionType.getName());
    return true;
  }

  bool traverseSubprogramType(const SubprogramType &subprogramTy) {
    {
      ColorScope color(OS, Cfg & Node::PrintConfig::Color, AddressColor);
      OS << "<";
    }
    llvm::interleaveComma(
        subprogramTy.getParamTypes(), OS,
        [&](const Type *type) { TypeBase::traverseType(*type); });
    {
      ColorScope color(OS, Cfg & Node::PrintConfig::Color, AddressColor);
      OS << ">";
    }
    OS << " -> ";
    TypeBase::traverseType(subprogramTy.getReturnType());
    return true;
  }
  //=--------------------------------------------------------------------------=//
  // End type printing functions
  //=--------------------------------------------------------------------------=//

private:
  raw_ostream &OS;
  Node::PrintConfig Cfg;
};

class StandaloneRecursiveHIRTypePrinter : protected TextTreeStructure,
                                          public RecursiveHIRTypePrinterBase {
public:
  StandaloneRecursiveHIRTypePrinter(raw_ostream &os)
      : TextTreeStructure(os, /*showColors=*/false),
        RecursiveHIRTypePrinterBase(os) {}
};

class RecursiveHIRPrinter : protected TextTreeStructure,
                            public RecursiveASTVisitor<RecursiveHIRPrinter>,
                            public RecursiveHIRTypePrinterBase {
  using HIRBase = RecursiveASTVisitor<RecursiveHIRPrinter>;

public:
  RecursiveHIRPrinter(raw_ostream &os, const SourceManager &sm,
                      Node::PrintConfig cfg)
      : TextTreeStructure(os, cfg & Node::PrintConfig::Color),
        RecursiveHIRTypePrinterBase(os, cfg), OS(os), SM(sm), Cfg(cfg) {}

  bool visitNode(const Node &node) {
    {
      ColorScope color(OS, Cfg & Node::Color, NodeColor);
      OS << ToString(node.getKind());
    }

    if (Cfg & Node::PrintConfig::Address) {
      PrintPointer(&node, Cfg & Node::PrintConfig::Color, AddressColor, OS);
    }

    if (Cfg & Node::PrintConfig::SourceLocation) {
      PrintSourceLocation({node.getBeginLoc(), node.getEndLoc()},
                          Cfg & Node::PrintConfig::Color, SourceLocationColor,
                          SM, OS);
    }
    OS << ':';

    HIRBase::visitNode(node);
    return true;
  }

  bool traverseNode(const Node &node) {
    AddChild([&] { HIRBase::traverseNode(node); });
    return true;
  }

  //=--------------------------------------------------------------------------=//
  // End node printing functions
  //=--------------------------------------------------------------------------=//
  bool visitCompilationUnit(const CompilationUnit &compUnit) { return true; }

  bool visitSubprogramDecl(const SubprogramDecl &subprogramDecl) {
    printAttribute(ToString(subprogramDecl.getFunctionKind()));
    printIdentifier(subprogramDecl.getName());

    OS << " ";

    traverseType(subprogramDecl.getType());
    return true;
  }

  bool visitParamDecl(const ParamDecl &paramDecl) {
    OS << " ";
    traverseType(paramDecl.getType());
    printIdentifier(paramDecl.getName());
    return true;
  }

  bool visitDataDecl(const DataDecl &dataDecl) {
    OS << " ";
    printIdentifier(dataDecl.getName());
    return true;
  }

  bool visitDataFieldDecl(const DataFieldDecl &dataFieldDecl) {
    OS << " ";
    traverseType(dataFieldDecl.getType());
    OS << " ";
    printIdentifier(dataFieldDecl.getName());
    return true;
  }

  bool visitPlaceholderDecl(const PlaceholderDecl &placeholderDecl) {
    OS << " ";
    traverseType(placeholderDecl.getType());
    OS << " :";
    printIdentifier(placeholderDecl.getName());
    return true;
  }

  bool visitExprIntegerNumber(const ExprIntegerNumber &exprNum) {
    OS << ' ';
    traverseType(exprNum.getType());
    OS << " :";
    printLiteral(exprNum.getNumber());
    return true;
  }

  bool visitExprTuple(const ExprTuple &exprTuple) {
    OS << ' ';
    traverseType(exprTuple.getType());
    return true;
  }

  bool visitExprRet(const ExprRet &exprRet) {
    OS << ' ';
    traverseType(exprRet.getType());
    return true;
  }

  bool visitExprVarRef(const ExprVarRef &exprVarRef) {
    OS << ' ' << exprVarRef.getName() << ' ';
    traverseType(exprVarRef.getType());
    return true;
  }

  bool visitExprAggregateDataAccess(
      const ExprAggregateDataAccess &exprDataFieldAcc) {
    if (auto name = exprDataFieldAcc.getName()) {
      OS << ' ' << *name;
    } else {
      OS << ' ' << exprDataFieldAcc.getIdxAccess();
    }

    OS << ' ';
    traverseType(exprDataFieldAcc.getType());
    return true;
  }

  bool visitExprMatch(const ExprMatch &exprMatch) {
    OS << ' ';
    traverseType(exprMatch.getType());
    return true;
  }

  bool visitExprMatchCase(const ExprMatchCase &matchCase) {
    OS << ' ';
    traverseType(matchCase.getType());
    return true;
  }

  bool visitAggregateDestructuration(const AggregateDestructuration &aggreDes) {
    OS << ' ';
    traverseType(aggreDes.getDestructuringType());
    return true;
  }

  bool visitAggregateDestructurationElem(
      const AggregateDestructurationElem &aggreDesElem) {
    OS << ' ';
    traverseType(aggreDesElem.getType());
    printIdentifier(" idx: ");
    OS << aggreDesElem.getIdxOfAggregateAccess();
    return true;
  }

  bool visitUnionAlternativeFieldDecl(
      const UnionAlternativeFieldDecl &enumAlternativeField) {
    OS << ' ';
    traverseType(enumAlternativeField.getType());
    OS << " :";
    printIdentifier(enumAlternativeField.getName());
    return true;
  }

  bool visitUnionAlternativeDecl(const UnionAlternativeDecl &alternative) {
    OS << ' ';
    traverseType(alternative.getType());
    return true;
  }

  bool visitUnionDecl(const UnionDecl &enumDecl) {
    OS << ' ';
    traverseType(enumDecl.getType());
    return true;
  }

  //=--------------------------------------------------------------------------=//
  // End node printing functions
  //=--------------------------------------------------------------------------=//

private:
  template <typename T> void printLiteral(T literal) {
    ColorScope color(OS, Cfg & Node::PrintConfig::Color, LiteralColor);
    OS << ' ' << literal;
  }

  void printAttribute(StringRef attr) {
    ColorScope color(OS, Cfg & Node::Color, AttrColor);
    OS << ' ' << attr;
  }

  void printIdentifier(StringRef id) {
    ColorScope color(OS, Cfg & Node::Color, IdentifierColor);
    OS << ' ' << id;
  }

private:
  raw_ostream &OS;
  const SourceManager &SM;
  const Node::PrintConfig Cfg;
};

} // namespace

void tmplang::hir::Node::print(raw_ostream &os, const SourceManager &sm,
                               PrintConfig cfg) const {
  RecursiveHIRPrinter(os, sm, cfg).traverseNode(*this);
}

void tmplang::hir::Node::dump(const SourceManager &sm,
                              Node::PrintConfig cfg) const {
  // FIXME: Set our own debug streams
  if (llvm::dbgs().has_colors()) {
    llvm::dbgs().enable_colors(true);
  }

  print(llvm::dbgs(), sm, cfg);

  llvm::dbgs().enable_colors(false);
}

void tmplang::hir::Type::print(llvm::raw_ostream &out) const {
  // TODO: Add colors
  StandaloneRecursiveHIRTypePrinter(out).traverseType(*this);
}

void tmplang::hir::Type::dump() const {
  // FIXME: Set our own debug streams
  print(llvm::dbgs());
}
