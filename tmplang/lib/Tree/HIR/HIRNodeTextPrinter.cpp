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
static constexpr TerminalColor SourceLocationColor = {raw_ostream::CYAN, true};
static constexpr TerminalColor NodeColor = {raw_ostream::GREEN, true};
static constexpr TerminalColor TypeColor = {raw_ostream::BLUE, true};

class RecursiveHIRPrinter : protected TextTreeStructure,
                            public RecursiveASTVisitor<RecursiveHIRPrinter>,
                            public RecursiveTypeVisitor<RecursiveHIRPrinter> {
public:
  using HIRBase = RecursiveASTVisitor<RecursiveHIRPrinter>;
  using TypeBase = RecursiveTypeVisitor<RecursiveHIRPrinter>;

  RecursiveHIRPrinter(raw_ostream &os, const SourceManager &sm,
                      Node::PrintConfig cfg)
      : TextTreeStructure(os, cfg & Node::PrintConfig::Color), OS(os), SM(sm),
        Cfg(cfg) {}

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

  bool visitFunctionDecl(const FunctionDecl &funcDecl) {
    printAttribute(ToString(funcDecl.getFunctionKind()));
    printIdentifier(funcDecl.getName());

    OS << " -> ";

    traverseType(funcDecl.getReturnType());
    return true;
  }

  bool visitParamDecl(const ParamDecl &paramDecl) {
    OS << " ";
    traverseType(paramDecl.getType());
    printIdentifier(paramDecl.getName());
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
  //=--------------------------------------------------------------------------=//
  // End type printing functions
  //=--------------------------------------------------------------------------=//

private:
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
