configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Version.inc.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Version.inc"
)

add_subdirectory(CLI)
add_subdirectory(Lowering)
