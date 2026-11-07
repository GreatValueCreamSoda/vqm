# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.

cmake_minimum_required(VERSION 3.25)
project(vqm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VQM_BUILD_CLI "Build the example CLI" ${PROJECT_IS_TOP_LEVEL})
option(VQM_ENABLE_BESTSOURCE "Enable the Linux BestSource Vulkan Video CLI backend" OFF)
option(VQM_BUILD_BENCH "Build benchmark tools (needs libpng/libjpeg)" ${VQM_BUILD_CLI})
option(VQM_BUILD_TESTS "Build the test suite" ${PROJECT_IS_TOP_LEVEL})
option(VQM_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(VQM_INSTALL "Install the core library and CMake package" ${PROJECT_IS_TOP_LEVEL})

if(VQM_BUILD_TESTS)
  include(CTest)
endif()

# One warning set for every target, so the core and the tools cannot drift.
add_library(vqm_warnings INTERFACE)
target_compile_options(vqm_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wsign-conversion>
  $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive->
  $<$<AND:$<BOOL:${VQM_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>
  $<$<AND:$<BOOL:${VQM_WARNINGS_AS_ERRORS}>,$<CXX_COMPILER_ID:MSVC>>:/WX>)

# The Vulkan compute backend is the only implementation. There is no CPU
# fallback and no MoltenVK path, so the build fails loudly instead of
# producing a library that cannot score anything.
if(APPLE)
  message(FATAL_ERROR
    "vqm requires Vulkan; Apple platforms are not supported (no MoltenVK path)")
endif()
find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_program(VQM_GLSLC NAMES glslc REQUIRED)
set(VQM_VULKAN_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(VQM_VULKAN_LIBRARIES Vulkan::Vulkan)
include(cmake/shaders.cmake)

set(VQM_CORE_SOURCES
  # Shared image input and orchestration.
  src/handles.cc
  src/image_view.cc
  src/input.cc
  src/pool.cc
  src/timing.cc
  # Metric composition.
  src/metrics/metrics.cc
  src/metrics/shared_workspace.cc
  # SSIMULACRA 2.
  src/metrics/ssimulacra2/pipeline.cc
  src/metrics/ssimulacra2/score.cc
  src/metrics/ssimulacra2/layout.cc
  src/metrics/ssimulacra2/vulkan.cc
  # Butteraugli.
  src/metrics/butteraugli/tables.cc
  src/metrics/butteraugli/pipeline.cc
  src/metrics/butteraugli/layout.cc
  src/metrics/butteraugli/vulkan.cc
  # PSNR-HVS.
  src/metrics/psnr_hvs/pipeline.cc
  src/metrics/psnr_hvs/layout.cc
  src/metrics/psnr_hvs/vulkan.cc
  # ColorVideoVDP.
  src/metrics/cvvdp/cvvdp_pipeline.cc
  src/metrics/cvvdp/cvvdp_display.cc
  src/metrics/cvvdp/cvvdp_model.cc
  src/metrics/cvvdp/cvvdp_spatial.cc
  # VMAF features, models and sequences.
  src/metrics/vmaf/features/vmaf_adm.cc
  src/metrics/vmaf/features/vmaf_preprocess.cc
  src/metrics/vmaf/features/vmaf_hdr.cc
  src/metrics/vmaf/features/vmaf_vif.cc
  src/metrics/vmaf/cambi/vmaf_cambi.cc
  src/metrics/vmaf/speed/vmaf_speed.cc
  src/metrics/vmaf/v0/vmaf_model.cc
  src/metrics/vmaf/v0/vmaf_pipeline.cc
  src/metrics/vmaf/v1/vmaf_v1_model.cc
  src/metrics/vmaf/v1/vmaf_v1_pipeline.cc
  # Metric-independent Vulkan runtime.
  src/vulkan/vulkan_support.cc
  src/vulkan/vulkan_source.cc
  src/vulkan/vulkan_memory_query.cc
  src/vulkan/vulkan_runtime.cc
  src/vulkan/vulkan_runtime_scratch.cc
  src/vulkan/vulkan_runtime_inputs.cc
  src/vulkan/vulkan_shared_storage.cc)

list(TRANSFORM VQM_CORE_SOURCES PREPEND "${PROJECT_SOURCE_DIR}/")

# Internal feature headers are shared by the core and its tests, not exported.
set(VQM_INTERNAL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")

# The core metrics take decoded pixel planes and link only the Vulkan
# loader.
function(vqm_configure_core target)
  target_sources(${target} PRIVATE ${VQM_CORE_SOURCES})
  add_dependencies(${target} vqm_shaders)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${VQM_INTERNAL_INCLUDE_DIRS})
  target_include_directories(${target} PRIVATE ${VQM_VULKAN_INCLUDE_DIRS})
  target_compile_features(${target} PUBLIC cxx_std_20)
  target_link_libraries(${target} PUBLIC ${VQM_VULKAN_LIBRARIES}
    PRIVATE vqm_warnings Threads::Threads)
endfunction()

add_library(vqm_core STATIC)
vqm_configure_core(vqm_core)
add_library(vqm::core ALIAS vqm_core)
set_target_properties(vqm_core PROPERTIES
  EXPORT_NAME core POSITION_INDEPENDENT_CODE ON)
set_target_properties(vqm_warnings PROPERTIES EXPORT_NAME warnings)

if(VQM_INSTALL)
  install(TARGETS vqm_core vqm_warnings
    EXPORT vqmTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES include/cvvdp.h include/psnr_hvs.h include/vqm_image.h include/ssimulacra2.h include/butteraugli.h include/vmaf.h include/vmaf_v1.h include/vqm.h include/vqm_alpha.h include/vqm_views.h
                include/vqm_vulkan.h include/vqm_device.h include/vqm_pool.h include/vqm_memory.h include/vqm_memory_types.h include/vqm_shared.h include/vqm_metrics.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(FILES LICENSE licenses/COLORVIDEOVDP-LICENSE licenses/JPEG-XL-LICENSE licenses/JPEG-XL-PATENTS licenses/NETFLIX-VMAF-LICENSE
                licenses/LIBSVM-LICENSE licenses/XIPH-PSNR-HVS-LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/vqm)
  set(vqm_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/vqm")
  configure_package_config_file(cmake/vqmConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/vqmConfig.cmake"
    INSTALL_DESTINATION "${vqm_cmake_dir}")
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/vqmConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
  install(EXPORT vqmTargets
    FILE vqmTargets.cmake NAMESPACE vqm:: DESTINATION "${vqm_cmake_dir}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vqmConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/vqmConfigVersion.cmake"
    DESTINATION "${vqm_cmake_dir}")
endif()

# File decoding (PNG/JPEG/Y4M) is not part of the core; the CLI, the bench and
# the tests share one library for it.
if(VQM_BUILD_CLI OR VQM_BUILD_BENCH OR
   (VQM_BUILD_TESTS AND BUILD_TESTING))
  find_package(PNG REQUIRED)
  find_package(JPEG REQUIRED)
  add_library(vqm_io STATIC tools/image_io/io.cc tools/cvvdp_options.cc)
  target_include_directories(vqm_io PRIVATE src include tools/image_io PUBLIC tools)
  target_compile_features(vqm_io PUBLIC cxx_std_20)
  target_link_libraries(vqm_io PRIVATE vqm_warnings
                                               PNG::PNG JPEG::JPEG)
endif()

# The command-line tools are separate consumers of the library.
if(VQM_BUILD_CLI)
  add_subdirectory(cli)
endif()

if(VQM_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(VQM_BUILD_TESTS AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
