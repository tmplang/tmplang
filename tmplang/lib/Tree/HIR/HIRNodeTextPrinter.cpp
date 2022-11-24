#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/Support/SourceManager.h>
#include <tmplang/Tree/HIR/RecursiveNodeVisitor.h>
#include <tmplang/Tree/HIR/RecursiveTypeVisitor.h>

using namespace tmplang;
using namespace tmplang::hir;

namespace {

struct TerminalColor {
  llvm::raw_ostream::Colors Color;
  bool Bold;
};

static constexpr TerminalColor IndentColor = {llvm::raw_ostream::WHITE, false};
static constexpr TerminalColor AddressColor = {llvm::raw_ostream::YELLOW,
                                               false};
static constexpr TerminalColor IdentifierColor = {llvm::raw_ostream::GREEN,
                                                  false};
static constexpr TerminalColor AttrColor = {llvm::raw_ostream::RED, false};
static constexpr TerminalColor SourceLocationColor = {llvm::raw_ostream::CYAN,
                                                      true};
static constexpr TerminalColor NodeColor = {llvm::raw_ostream::GREEN, true};
static constexpr TerminalColor TypeColor = {llvm::raw_ostream::BLUE, true};

class ColorScope {
  llvm::raw_ostream &O;
  bool ShowColors;

public:
  ColorScope(llvm::raw_ostream &O, Node::PrintConfig cfg, TerminalColor color)
      : O(O), ShowColors(cfg & Node::PrintConfig::Color) {
    if (ShowColors) {
      O.changeColor(color.Color, color.Bold);
    }
  }
  ~ColorScope() {
    if (ShowColors) {
      O.resetColor();
    }
  }
};

//=--------------------------------------------------------------------------=//
// TextTreeStructure - Helper class to keep a tree-like indentation
//=--------------------------------------------------------------------------=//
// FROM: clang/include/clang/AST/TextNodeDumper.h:33
//=--------------------------------------------------------------------------=//
class TextTreeStructure {
  llvm::raw_ostream &OS;
  const Node::PrintConfig Cfg;

  /// Pending[i] is an action to dump an entity at level i.
  llvm::SmallVector<std::function<void(bool IsLastChild)>, 32> Pending;

  /// Indicates whether we're at the top level.
  bool TopLevel = true;

  /// Indicates if we're handling the first child after entering a new depth.
  bool FirstChild = true;

  /// Prefix for currently-being-dumped entity.
  std::string Prefix;

public:
  /// Add a child of the current node.  Calls DoAddChild without arguments
  template <typename Fn> void AddChild(Fn DoAddChild) {
    return AddChild("", DoAddChild);
  }

  /// Add a child of the current node with an optional label.
  /// Calls DoAddChild without arguments.
  template <typename Fn> void AddChild(llvm::StringRef Label, Fn DoAddChild) {
    // If we're at the top level, there's nothing interesting to do; just
    // run the dumper.
    if (TopLevel) {
      TopLevel = false;
      DoAddChild();
      while (!Pending.empty()) {
        Pending.back()(true);
        Pending.pop_back();
      }
      Prefix.clear();
      OS << "\n";
      TopLevel = true;
      return;
    }

    auto DumpWithIndent = [this, DoAddChild,
                           Label(Label.str())](bool IsLastChild) {
      // Print out the appropriate tree structure and work out the prefix for
      // children of this node. For instance:
      //
      //   A        Prefix = ""
      //   |-B      Prefix = "| "
      //   | `-C    Prefix = "|   "
      //   `-D      Prefix = "  "
      //     |-E    Prefix = "  | "
      //     `-F    Prefix = "    "
      //   G        Prefix = ""
      //
      // Note that the first level gets no prefix.
      {
        OS << '\n';
        ColorScope Color(OS, Cfg, IndentColor);
        OS << Prefix << (IsLastChild ? '`' : '|') << '-';
        if (!Label.empty())
          OS << Label << ": ";

        this->Prefix.push_back(IsLastChild ? ' ' : '|');
        this->Prefix.push_back(' ');
      }

      FirstChild = true;
      unsigned Depth = Pending.size();

      DoAddChild();

      // If any children are left, they're the last at their nesting level.
      // Dump those ones out now.
      while (Depth < Pending.size()) {
        Pending.back()(true);
        this->Pending.pop_back();
      }

      // Restore the old prefix.
      this->Prefix.resize(Prefix.size() - 2);
    };

    if (FirstChild) {
      Pending.push_back(std::move(DumpWithIndent));
    } else {
      Pending.back()(false);
      Pending.back() = std::move(DumpWithIndent);
    }
    FirstChild = false;
  }

  TextTreeStructure(llvm::raw_ostream &OS, Node::PrintConfig cfg)
      : OS(OS), Cfg(cfg) {}
};

class RecursiveHIRPrinter : protected TextTreeStructure,
                            public RecursiveASTVisitor<RecursiveHIRPrinter>,
                            public RecursiveTypeVisitor<RecursiveHIRPrinter> {
public:
  using HIRBase = RecursiveASTVisitor<RecursiveHIRPrinter>;
  using TypeBase = RecursiveTypeVisitor<RecursiveHIRPrinter>;

  RecursiveHIRPrinter(llvm::raw_ostream &os, const SourceManager &sm,
                      Node::PrintConfig cfg)
      : TextTreeStructure(os, /*showColors=*/cfg & Node::PrintConfig::Color),
        OS(os), SM(sm), Cfg(cfg) {}

  bool visitNode(const Node &node) {
    {
      ColorScope color(OS, Cfg, NodeColor);
      OS << ToString(node.getKind());
    }

    bool printAddress = Cfg & Node::PrintConfig::Address;
    bool printSrcLoc = Cfg & Node::PrintConfig::SourceLocation;

    if (printAddress || printSrcLoc) {
      if (printAddress) {
        printPointer(&node);
      }
      if (printSrcLoc) {
        printSourceLocation(node);
      }
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
    ColorScope color(OS, Cfg, TypeColor);
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
  void printSourceLocation(const Node &node) {
    ColorScope color(OS, Cfg, SourceLocationColor);

    const LineAndColumn begin = SM.getLineAndColumn(node.getBeginLoc());
    const LineAndColumn end = SM.getLineAndColumn(node.getEndLoc());

    OS << llvm::formatv(" <{0},{1}-{2},{3}>", begin.Line, begin.Column,
                        end.Line, end.Column);
  }

  void printPointer(const void *ptr) {
    ColorScope color(OS, Cfg, AddressColor);
    OS << ' ' << ptr;
  }

  void printAttribute(llvm::StringRef attr) {
    ColorScope color(OS, Cfg, AttrColor);
    OS << ' ' << attr;
  }

  void printIdentifier(llvm::StringRef id) {
    ColorScope color(OS, Cfg, IdentifierColor);
    OS << ' ' << id;
  }

private:
  llvm::raw_ostream &OS;
  const SourceManager &SM;
  const Node::PrintConfig Cfg;
};

} // namespace

void tmplang::hir::Node::print(llvm::raw_ostream &os, const SourceManager &sm,
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
