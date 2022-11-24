// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

def General_Group : OptionGroup<"<general group>">, HelpText<"Options">;
def Dump_Group : OptionGroup<"<dump group>">, HelpText<"Dump options">;

// General options
def version : Flag<["--"], "version">, Group<General_Group>,
              HelpText<"print version">;
def v : Flag<["-"], "v">, Group<General_Group>, Alias<version>;
def help : Flag<["--"], "help">, Group<General_Group>, HelpText<"print help">;
def h : Flag<["-"], "h">, Group<General_Group>, Alias<help>;

// Dump options
def dump_hir: CommaJoined<["--"], "dump-hir=">, Group<Dump_Group>,
              HelpText<"multi options: color|addr|loc|all|simple\nDefault: all">;

def Diagnostics_Group : OptionGroup<"<diagnostics group>">, HelpText<"Diagnostics options">;

// Diagnostics options
def color_diagnostics : Flag<["--"], "color-diagnostics">, Group<Diagnostics_Group>,
                        HelpText<"prints colorful diagnostics">;
