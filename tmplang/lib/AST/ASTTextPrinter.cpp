#include <llvm/Support/raw_ostream.h>
#include <tmplang/AST/RecursiveASTVisitor.h>
#include <tmplang/AST/RecursiveTypeVisitor.h>

using namespace tmplang;

namespace {

struct TerminalColor {
  llvm::raw_ostream::Colors Color;
  bool Bold;
};
static const TerminalColor IndentColor = {llvm::raw_ostream::BLUE, false};

class ColorScope {
  llvm::raw_ostream &O;
  bool ShowColors;

public:
  ColorScope(llvm::raw_ostream &O, bool showColors, TerminalColor color)
      : O(O), ShowColors(showColors) {
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
  const bool ShowColors;

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
        ColorScope Color(OS, ShowColors, IndentColor);
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

  TextTreeStructure(llvm::raw_ostream &OS, bool ShowColors)
      : OS(OS), ShowColors(ShowColors) {}
};

class RecursiveASTPrinter : protected TextTreeStructure,
                            public RecursiveASTVisitor<RecursiveASTPrinter>,
                            public RecursiveTypeVisitor<RecursiveASTPrinter> {
public:
  using ASTBase = RecursiveASTVisitor<RecursiveASTPrinter>;
  using TypeBase = RecursiveTypeVisitor<RecursiveASTPrinter>;

  RecursiveASTPrinter(llvm::raw_ostream &os)
      : TextTreeStructure(os, /*showColors=*/false), OS(os) {}

  //=--------------------------------------------------------------------------=//
  // End node printing functions
  //=--------------------------------------------------------------------------=//
  bool visitCompilationUnit(const CompilationUnit &compUnit) {
    OS << "CompilationUnit";
    printPointer(&compUnit);

    llvm::for_each(compUnit.getDecls(), [=](const std::unique_ptr<Decl> &decl) {
      AddChild([&]() { visitNode(*decl); });
    });

    return true;
  }

  bool visitFuncDecl(const FunctionDecl &funcDecl) {
    OS << "FuncDecl";
    printPointer(&funcDecl);

    // TODO: Print function kind, not yet defined
    OS << " " << funcDecl.getName();
    OS << " ->";
    traverseType(funcDecl.getReturnType());

    llvm::for_each(funcDecl.getParams(), [=](const ParamDecl &paramDecl) {
      AddChild([&]() { visitNode(paramDecl); });
    });

    return true;
  }

  bool visitParamDecl(const ParamDecl &paramDecl) {
    OS << "ParamDecl";
    printPointer(&paramDecl);
    traverseType(paramDecl.getType());

    return true;
  }
  //=--------------------------------------------------------------------------=//
  // End node printing functions
  //=--------------------------------------------------------------------------=//

  //=--------------------------------------------------------------------------=//
  // Begin type printing functions
  //=--------------------------------------------------------------------------=//
  bool visitBuiltinType(const BuiltinType &builtinType) {
    OS << " " << ToString(builtinType.getBuiltinKind());
    return true;
  }
  //=--------------------------------------------------------------------------=//
  // End type printing functions
  //=--------------------------------------------------------------------------=//

private:
  void printPointer(const void *ptr) { OS << ' ' << ptr; }

private:
  llvm::raw_ostream &OS;
};

} // namespace
