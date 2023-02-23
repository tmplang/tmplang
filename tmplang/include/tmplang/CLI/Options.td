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

// Output options
def Output_Group : OptionGroup<"<output group>">, HelpText<"Output options">;
def output: JoinedOrSeparate<["--"], "output">, Group<Output_Group>,
              HelpText<"output name of the generated assembly file">;
def o : JoinedOrSeparate<["-"], "o">, Group<Output_Group>, Alias<output>;

// Dump options
def dump_src: CommaJoined<["--"], "dump-src=">, Group<Dump_Group>,
              HelpText<"multi options: color|addr|loc|all|simple">;
def dump_hir: CommaJoined<["--"], "dump-hir=">, Group<Dump_Group>,
              HelpText<"multi options: color|addr|loc|all|simple">;
def dump_mlir: CommaJoined<["--"], "dump-mlir=">, Group<Dump_Group>,
              HelpText<"multi options: lower|opt|trans|llvm|all">;

def Diagnostics_Group : OptionGroup<"<diagnostics group>">, HelpText<"Diagnostics options">;

// Diagnostics options
def color_diagnostics : Flag<["--"], "color-diagnostics">, Group<Diagnostics_Group>,
                        HelpText<"prints colorful diagnostics">;
