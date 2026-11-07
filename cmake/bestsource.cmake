# Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

find_package(PkgConfig REQUIRED)
pkg_check_modules(VQM_BESTSOURCE_FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=63.1.0 libavcodec>=63.1.0 libavutil>=61.6.0)
pkg_check_modules(VQM_BESTSOURCE_XXHASH REQUIRED IMPORTED_TARGET libxxhash)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(FetchContent)
# BestSource R23 and the tested revision of its libp2p fallback. Follow the
# source lists and GPU shader definitions in R23's Meson build and libp2p wrap.
FetchContent_Declare(vqm_bestsource
  URL https://codeload.github.com/vapoursynth/bestsource/tar.gz/bca32dac44794d93f3e580f9573c58c4a0e7671f
  URL_HASH SHA256=400b4d89232550018fefd7eb0ef0c018bc15a4b427c84d6d77d81e88574e6000
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
FetchContent_Declare(vqm_libp2p
  URL https://codeload.github.com/sekrit-twc/libp2p/tar.gz/869fa993041f9f3af7d9ac8b10158920c6ddce66
  URL_HASH SHA256=cacef2683a19a8b288cf567544b507f7f06dd66160db566bf3349e5ee0b73d90
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
FetchContent_MakeAvailable(vqm_bestsource vqm_libp2p)

add_library(vqm_libp2p STATIC
  "${vqm_libp2p_SOURCE_DIR}/p2p_api.cpp"
  "${vqm_libp2p_SOURCE_DIR}/v210.cpp"
  "${vqm_libp2p_SOURCE_DIR}/simd/cpuinfo_x86.cpp"
  "${vqm_libp2p_SOURCE_DIR}/simd/p2p_simd.cpp")
target_include_directories(vqm_libp2p SYSTEM PUBLIC "${vqm_libp2p_SOURCE_DIR}")
set_target_properties(vqm_libp2p PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED ON)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86|x86)$")
  set(p2p_sse41 "${vqm_libp2p_SOURCE_DIR}/simd/p2p_sse41.cpp")
  target_sources(vqm_libp2p PRIVATE "${p2p_sse41}")
  target_compile_definitions(vqm_libp2p PRIVATE P2P_SIMD)
  set_source_files_properties("${p2p_sse41}" PROPERTIES COMPILE_OPTIONS -msse4.1)
endif()

set(bestsource_src "${vqm_bestsource_SOURCE_DIR}/src")
set(bestsource_shader "${bestsource_src}/shaders/hashexport.comp.glsl")
set(bestsource_samples_x 8)
foreach(bits 8 16)
  set(spv "${CMAKE_CURRENT_BINARY_DIR}/hashexport${bits}.spv")
  add_custom_command(
    OUTPUT "${spv}"
    COMMAND "${VQM_GLSLC}" --target-env=vulkan1.2 -fshader-stage=compute
            "-DSAMPLE_BITS=${bits}" "-DSAMPLES_X=${bestsource_samples_x}"
            "${bestsource_shader}" -o "${spv}"
    DEPENDS "${bestsource_shader}"
    VERBATIM)
  set(bestsource_spv${bits} "${spv}")
endforeach()

set(bestsource_shader_header "${CMAKE_CURRENT_BINARY_DIR}/hashexport_spv.h")
set(bestsource_spv2c "${vqm_bestsource_SOURCE_DIR}/tools/spv2c.py")
add_custom_command(
  OUTPUT "${bestsource_shader_header}"
  COMMAND "${Python3_EXECUTABLE}" "${bestsource_spv2c}" "${bestsource_shader_header}"
          "BSHashExportSpv8=${bestsource_spv8}" "BSHashExportSpv16=${bestsource_spv16}"
  DEPENDS "${bestsource_spv2c}" "${bestsource_spv8}" "${bestsource_spv16}"
  VERBATIM)

add_library(vqm_bestsource STATIC
  "${bestsource_src}/audiosource.cpp"
  "${bestsource_src}/bsshared.cpp"
  "${bestsource_src}/gpuhash.cpp"
  "${bestsource_src}/tracklist.cpp"
  "${bestsource_src}/videosource.cpp"
  "${bestsource_shader_header}")
target_include_directories(vqm_bestsource SYSTEM PUBLIC "${bestsource_src}"
  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
# R23 uses the C++17 std::string overloads of filesystem's UTF-8 API.
set_target_properties(vqm_bestsource PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(vqm_bestsource PUBLIC BS_GPU=1
  PRIVATE "BS_GPU_SAMPLES_X=${bestsource_samples_x}")
target_link_libraries(vqm_bestsource PUBLIC PkgConfig::VQM_BESTSOURCE_FFMPEG
  PRIVATE vqm_libp2p PkgConfig::VQM_BESTSOURCE_XXHASH Vulkan::Headers Threads::Threads)

if(VQM_INSTALL)
  install(FILES "${vqm_bestsource_SOURCE_DIR}/LICENSE"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/vqm RENAME BESTSOURCE-LICENSE)
  install(FILES "${vqm_libp2p_SOURCE_DIR}/COPYING"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/vqm RENAME LIBP2P-LICENSE)
endif()
