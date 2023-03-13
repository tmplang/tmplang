llvm_add_library(TmplangCodegen
    Codegen.cpp

    LINK_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    Analysis # Needed for TargetLibraryInfo
    Codegen
    Core
    MC
    Target
    Support
    Option
)
