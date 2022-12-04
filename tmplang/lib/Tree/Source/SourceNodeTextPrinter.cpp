#include <tmplang/Tree/Source/RecursiveNodeVisitor.h>
#include <tmplang/Tree/Source/RecursiveTypeVisitor.h>

#include <llvm/Support/Debug.h>

#include "../TreeFormatPrinter.h"
#include "llvm/ADT/Optional.h"

using namespace tmplang;
using namespace tmplang::source;

namespace {

static constexpr TerminalColor AddressColor = {llvm::raw_ostream::YELLOW,
                                               false};
static constexpr TerminalColor IdentifierColor = {llvm::raw_ostream::GREEN,
                                                  false};
static constexpr TerminalColor ErrorRecoveryColor = {llvm::raw_ostream::RED,
                                                     false};
static constexpr TerminalColor SourceLocationColor = {llvm::raw_ostream::CYAN,
                                                      true};
static constexpr TerminalColor NodeColor = {llvm::raw_ostream::GREEN, true};
static constexpr TerminalColor TypeColor = {llvm::raw_ostream::BLUE, true};

class RecursiveSourcePrinter
    : protected TextTreeStructure,
      public RecursiveASTVisitor<RecursiveSourcePrinter>,
      public RecursiveTypeVisitor<RecursiveSourcePrinter> {
public:
  using SrcBase = RecursiveASTVisitor<RecursiveSourcePrinter>;
  using TypeBase = RecursiveTypeVisitor<RecursiveSourcePrinter>;

  RecursiveSourcePrinter(llvm::raw_ostream &os, const SourceManager &sm,
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

  bool visitFunctionDecl(const FunctionDecl &funcDecl) {
    printToken(funcDecl.getFuncType());
    OS << ' ';
    printToken(funcDecl.getIdentifier());
    if (auto &colon = funcDecl.getColon()) {
      OS << ' ';
      printToken(*colon);
    }
    if (auto arrow = funcDecl.getArrow()) {
      OS << ' ';
      printToken(*arrow);
    }
    if (auto *retType = funcDecl.getReturnType()) {
      OS << ' ';
      traverseType(*retType);
    }
    OS << ' ';
    printToken(funcDecl.getLKeyBracket());
    // FIXME: Find a proper location for the right key bracket once we have a
    // body
    OS << ' ';
    printToken(funcDecl.getRKeyBracket());
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

    llvm::ArrayRef<RAIIType> types = tupleType.getTypes();
    llvm::ArrayRef<Token> commas = tupleType.getCommas();

    for (unsigned i = 0; i < commas.size(); i++) {
      const RAIIType &type = types[i];
      const Token &comma = commas[i];

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
  llvm::raw_ostream &OS;
  const SourceManager &SM;
  const Node::PrintConfig Cfg;
};

} // namespace

void tmplang::source::Node::print(llvm::raw_ostream &os,
                                  const SourceManager &sm,
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
