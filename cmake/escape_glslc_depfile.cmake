# glslc emits absolute include paths without Make-style escaping. Escape the
# common source-root prefix so depfiles remain valid when a checkout path
# contains spaces, hashes or dollar signs. Shader paths below that root are
# controlled by this project. The target is relative to the build directory.
if(NOT DEFINED DEPFILE OR NOT DEFINED SOURCE_ROOT)
  message(FATAL_ERROR "DEPFILE and SOURCE_ROOT are required")
endif()

file(READ "${DEPFILE}" content)
set(escaped "${SOURCE_ROOT}")
string(REPLACE "\\" "\\\\" escaped "${escaped}")
string(REPLACE "$" "$$" escaped "${escaped}")
string(REPLACE "#" "\\#" escaped "${escaped}")
string(REPLACE " " "\\ " escaped "${escaped}")
string(REPLACE "${SOURCE_ROOT}/" "${escaped}/" content "${content}")
file(WRITE "${DEPFILE}" "${content}")
