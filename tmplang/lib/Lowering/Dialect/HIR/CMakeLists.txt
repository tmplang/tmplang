add_mlir_dialect_library(TmplangMLIRDialect
  TmplangDialect.cpp
  TmplangEnumAttrs.cpp
  TmplangOps.cpp
  TmplangTraits.cpp
  TmplangTypes.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  MLIRIR
  MLIRAnalysis
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms

  LINK_COMPONENTS
  Support
)

add_dependencies(TmplangMLIRDialect 
  MLIRTmplangHIROpsIncGen
  MLIRTmplangHIROpsEnumIncGen
  MLIRTmplangHIROpsAttributesIncGen
  MLIRBuiltinTypesIncGen
)
