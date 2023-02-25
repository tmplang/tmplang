#include <tmplang/CLI/Arguments.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/raw_ostream.h>
#include <tmplang/CLI/CLPrinter.h>
#include <tmplang/Version.inc>

using namespace tmplang;

namespace {

#define PREFIX(NAME, VALUE) static const char *const NAME[] = VALUE;
#include <tmplang/CLI/Options.inc>
#undef PREFIX

static constexpr llvm::opt::OptTable::Info TmplangInfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {PREFIX,      NAME,      HELPTEXT,                                           \
   METAVAR,     OPT_##ID,  llvm::opt::Option::KIND##Class,                     \
   PARAM,       FLAGS,     OPT_##GROUP,                                        \
   OPT_##ALIAS, ALIASARGS, VALUES},
#include <tmplang/CLI/Options.inc>
#undef OPTION
};

struct TmplLangOptTable : public llvm::opt::OptTable {
  TmplLangOptTable() : OptTable(TmplangInfoTable) {}
};

} // namespace

const llvm::opt::OptTable &tmplang::GetOptionsTable() {
  static const TmplLangOptTable Table;
  return Table;
}

std::unique_ptr<llvm::opt::InputArgList>
tmplang::ParseArgs(ArrayRef<const char *> rawArgs, CLPrinter &printer) {
  unsigned missingArgIndex;
  unsigned missingArgCount;
  auto result = std::make_unique<llvm::opt::InputArgList>(
      tmplang::GetOptionsTable().ParseArgs(rawArgs, missingArgIndex,
                                           missingArgCount));

  if (missingArgCount) {
    printer.errs() << "argument to '" << result->getArgString(missingArgIndex)
                   << "' is missing (expected " << missingArgCount
                   << (missingArgCount == 1 ? " value" : " values") << ")\n";
    return nullptr;
  }

  return result;
}

// Get the first alias found for the given option, or nullopt otherwise
static std::optional<llvm::opt::Option>
GetFirstAlias(const llvm::opt::OptTable &table, llvm::opt::Option o) {
  for (unsigned i = 0; i <= table.getNumOptions(); ++i) {
    llvm::opt::Option alias = table.getOption(i);
    if (!alias.isValid() || alias.getID() == o.getID()) {
      continue;
    }

    llvm::opt::Option aliasOf = alias.getAlias();
    if (aliasOf.isValid() && aliasOf.getID() == o.getID()) {
      return alias;
    }
  }
  return nullopt;
}

// Build the first column of the help output
static SmallString<80> BuildArg(llvm::opt::Option o,
                                const llvm::opt::OptTable &table) {
  SmallString<80> result;
  llvm::raw_svector_ostream out(result);

  // Find a short alias to print it before
  if (std::optional<llvm::opt::Option> alias = GetFirstAlias(table, o)) {
    llvm::formatv("{0} [ {1} ]", alias->getPrefixedName(), o.getPrefixedName())
        .format(out);
  } else {
    llvm::formatv("{0}", o.getPrefixedName()).format(out);
  }

  // Print the metavar if exists
  if (const char *metavar = table.getOptionMetaVar(o.getID())) {
    llvm::formatv(" {0}", metavar).format(out);
  }

  return result;
}

// Return true if the option is invalid, is a group, doesn't have prefix or is
// an alias of other option
static bool InvalidOption(llvm::opt::Option o) {
  return !o.isValid() || o.getKind() == llvm::opt::Option::GroupClass ||
         o.getPrefix().empty() || o.getUnaliasedOption().getID() != o.getID();
}

static llvm::SetVector<unsigned>
CollectAllGroups(const llvm::opt::OptTable &table) {
  // Collect all used groups
  llvm::SetVector<unsigned> groups;
  for (unsigned i = 0; i <= table.getNumOptions(); ++i) {
    llvm::opt::Option o = table.getOption(i);
    if (InvalidOption(o)) {
      continue;
    }

    llvm::opt::Option grp = o.getGroup();
    if (grp.isValid()) {
      groups.insert(grp.getID());
    }
  }

  return groups;
}

static SmallVector<StringRef, 8> WordWrap(StringRef text, unsigned col) {
  SmallVector<StringRef, 8> result;

  do {
    result.emplace_back(text.begin(), 0);

    StringRef line;
    std::tie(line, text) = text.split('\n');
    do {
      StringRef word;
      std::tie(word, line) = line.split(' ');

      unsigned length = word.end() - result.back().begin();
      if (length <= col) {
        result.back() = StringRef(result.back().begin(), length);
      } else {
        result.push_back(word);
      }
    } while (line.size());
  } while (text.size());

  return result;
}

static void PrintGroupHelp(const llvm::opt::OptTable &table, raw_ostream &out,
                           llvm::opt::OptSpecifier groupID) {
  out << table.getOptionHelpText(groupID) << ":\n";

  for (unsigned i = 0; i <= table.getNumOptions(); ++i) {
    llvm::opt::Option o = table.getOption(i);
    if (InvalidOption(o)) {
      continue;
    }
    // Skip options of other groups
    if (!o.matches(groupID)) {
      continue;
    }

    StringRef helpText;
    if (const char *text = table.getOptionHelpText(o.getID())) {
      helpText = text;
    }

    SmallString<80> arg = BuildArg(o, table);
    for (StringRef line : WordWrap(helpText, 47)) {
      // Break the line if both columns don't fit together
      if (arg.size() > 30) {
        llvm::formatv("  {0}\n", arg).format(out);
        arg.clear();
      }
      llvm::formatv("  {0,-30} {1}\n", arg, line).format(out);
      arg.clear();
    }
  }
}

static void PrintHelp(const llvm::opt::OptTable &table, CLPrinter &printer) {
  // Preamble
  printer.outs() << llvm::formatv("Syntax: {0} [options] <input>\n",
                                  printer.getExecName());

  // Print arguments grouped by group
  for (unsigned group : CollectAllGroups(table)) {
    printer.outs() << '\n';
    PrintGroupHelp(table, printer.outs(), group);
  }
}

static void PrintVersion(CLPrinter &printer) {
  printer.outs() << TMPLANG_VERSION_STR << '\n';
}

bool tmplang::HandleImmediateArgs(const llvm::opt::InputArgList &parsedArgs,
                                  CLPrinter &printer) {

  const llvm::opt::OptTable &table = tmplang::GetOptionsTable();
  if (parsedArgs.hasArg(tmplang::OPT_help)) {
    PrintHelp(table, printer);
    return true;
  }

  if (parsedArgs.hasArg(tmplang::OPT_version)) {
    PrintVersion(printer);
    return true;
  }

  return false;
}
