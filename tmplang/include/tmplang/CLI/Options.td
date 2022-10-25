// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

def General_Group : OptionGroup<"<general group>">, HelpText<"Options">;

def version : Flag<["--"], "version">, Group<General_Group>,
              HelpText<"print version">;
def v : Flag<["-"], "v">, Group<General_Group>, Alias<version>;
def help : Flag<["--"], "help">, Group<General_Group>, HelpText<"print help">;
def h : Flag<["-"], "h">, Group<General_Group>, Alias<help>;

def Dump_Group : OptionGroup<"<dump group>">, HelpText<"All dump options">;

def dump_hir: CommaJoined<["--"], "dump-hir=">, Group<General_Group>,
              HelpText<"multi options: color|addr|loc|all|simple\nDefault: all">;
