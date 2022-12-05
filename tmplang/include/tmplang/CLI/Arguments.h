#ifndef TMPLANG_CLI_ARGUMENTS_H
#define TMPLANG_CLI_ARGUMENTS_H

#include <llvm/ADT/ArrayRef.h>
#include <tmplang/ADT/LLVM.h>

namespace llvm::opt {
class OptTable;
class InputArgList;
} // namespace llvm::opt

namespace tmplang {

class CLPrinter;

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include <tmplang/CLI/Options.inc>
  LastOption
#undef OPTION
};

const llvm::opt::OptTable &GetOptionsTable();

/// Given a list of raw arguments, parse them and return as result the parsed
/// arguments. If any argument is missing, an error is printed on \ref errOut
/// and nullptr is returned
std::unique_ptr<llvm::opt::InputArgList>
ParseArgs(ArrayRef<const char *> rawArgs, CLPrinter &);

/// Checks for inmmediate args suck as --help or --version. If true is returned
/// any of both options were found and got printed on \ref outs
bool HandleImmediateArgs(const llvm::opt::InputArgList &parsedArgs,
                         CLPrinter &);

} // namespace tmplang

#endif // TMPLANG_CLI_ARGUMENTS_H
