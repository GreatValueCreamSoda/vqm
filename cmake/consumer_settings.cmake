# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.

# Keep the package smoke test on the toolchain and ABI used by its static core.
# An initial-cache file preserves list-valued settings and quoted flags without
# passing their contents through another command-line/list expansion.
function(vqm_write_consumer_settings output)
  set(variables
    CMAKE_CXX_COMPILER CMAKE_CXX_COMPILER_ARG1
    CMAKE_CXX_COMPILER_TARGET CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN
    CMAKE_CXX_COMPILER_LAUNCHER CMAKE_TOOLCHAIN_FILE
    CMAKE_SYSROOT CMAKE_SYSROOT_COMPILE CMAKE_SYSROOT_LINK
    CMAKE_FIND_ROOT_PATH CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
    CMAKE_FIND_ROOT_PATH_MODE_LIBRARY CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
    CMAKE_FIND_ROOT_PATH_MODE_PACKAGE CMAKE_PREFIX_PATH
    CMAKE_CXX_STANDARD_LIBRARIES CMAKE_MSVC_RUNTIME_LIBRARY CMAKE_BUILD_TYPE
    CMAKE_CONFIGURATION_TYPES CMAKE_GENERATOR_INSTANCE CMAKE_MAKE_PROGRAM)
  # Test-driver metadata uses the same quoting as compiler paths and flags.
  set(VQM_CONSUMER_GENERATOR "${CMAKE_GENERATOR}")
  set(VQM_CONSUMER_GENERATOR_PLATFORM "${CMAKE_GENERATOR_PLATFORM}")
  set(VQM_CONSUMER_GENERATOR_TOOLSET "${CMAKE_GENERATOR_TOOLSET}")
  set(VQM_CONSUMER_CROSSCOMPILING "${CMAKE_CROSSCOMPILING}")
  list(APPEND variables VQM_CONSUMER_GENERATOR VQM_CONSUMER_GENERATOR_PLATFORM
    VQM_CONSUMER_GENERATOR_TOOLSET VQM_CONSUMER_CROSSCOMPILING)
  get_cmake_property(parent_variables VARIABLES)
  foreach(variable IN LISTS parent_variables)
    if(variable MATCHES "^CMAKE_(CXX_FLAGS|EXE_LINKER_FLAGS)(_|$)")
      list(APPEND variables "${variable}")
    endif()
  endforeach()
  set(content "# Generated from the parent VQM build; do not edit.\n")
  foreach(variable IN LISTS variables)
    if(NOT DEFINED ${variable})
      continue()
    endif()
    set(value "${${variable}}")
    set(equals "=")
    string(FIND "${value}" "]${equals}]" closing_bracket)
    while(NOT closing_bracket EQUAL -1)
      string(APPEND equals "=")
      string(FIND "${value}" "]${equals}]" closing_bracket)
    endwhile()
    if(variable MATCHES "^VQM_CONSUMER_")
      string(APPEND content "set(${variable} [${equals}[${value}]${equals}])\n")
    else()
      string(APPEND content
        "set(${variable} [${equals}[${value}]${equals}] CACHE STRING \"Inherited from VQM\" FORCE)\n")
    endif()
  endforeach()
  file(WRITE "${output}" "${content}")
endfunction()
