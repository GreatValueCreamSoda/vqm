# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.

cmake_minimum_required(VERSION 3.25)
project(vqm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VQM_BUILD_CLI "Build the example CLI" ${PROJECT_IS_TOP_LEVEL})
set(VQM_ENABLE_NVDEC AUTO CACHE STRING "NVIDIA hardware video decoding in the Linux CLI (AUTO, ON, OFF)")
set_property(CACHE VQM_ENABLE_NVDEC PROPERTY STRINGS AUTO ON OFF)
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
# The plane shaders (prepare, scale) are built twice: once as f32 and once
# with SS2_HALF_PLANES for Ssimulacra2Config::half_planes
# selects. Only the SPIR-V blob names differ; there is one source each. reduce
# touches f32 norm sums only and stays single.
#
# scale.comp is the one shader whose geometry follows the device: it is built
# once per tile shape (see kScaleVariants in metrics/ssimulacra2/layout.cc;
# order and membership change together) and the pipeline runs the first
# variant the device's limits allow. Each entry is
# "rows cols lanes": the local size is 16 x lanes, so the invocations per
# workgroup are 16 * lanes, and the shared footprint is
# (2*(cols+12)*(rows+12) + (rows+12)*cols) * 4 bytes.
set(VQM_SCALE_VARIANTS
  "32 32 16"
  "16 32 8")
set(VQM_SHADER_OUTPUTS "")
set(VQM_SHADER_BLOBS "")

# vqm_add_shader(<blob name> <source file> [extra glslc flags...])
function(vqm_add_shader name source)
  set(spv "${CMAKE_CURRENT_BINARY_DIR}/shaders/${name}.spv")
  set(depfile "${spv}.d")
  add_custom_command(
    OUTPUT "${spv}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND ${VQM_GLSLC} -O --target-env=vulkan1.1
            -fshader-stage=compute ${ARGN}
            -I "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan/shaders"
            -MD -MF "${depfile}" -MT "shaders/${name}.spv"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/${source}"
            -o "${spv}"
    COMMAND ${CMAKE_COMMAND}
            "-DDEPFILE=${depfile}" "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/escape_glslc_depfile.cmake"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${source}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/escape_glslc_depfile.cmake"
    DEPFILE "${depfile}"
    COMMENT "glslc ${source} -> ${name}.spv"
    VERBATIM)
  list(APPEND VQM_SHADER_OUTPUTS "${spv}")
  list(APPEND VQM_SHADER_BLOBS "${name}")
  set(VQM_SHADER_OUTPUTS "${VQM_SHADER_OUTPUTS}" PARENT_SCOPE)
  set(VQM_SHADER_BLOBS "${VQM_SHADER_BLOBS}" PARENT_SCOPE)
endfunction()

# prepare and Butteraugli's opsin use the same source classes
# (src/vulkan/vulkan_support.h, same order and membership): chroma 0 = 4:4:4/GREY,
# 1 = subsampled; transfer 0 = descriptor chain, 1 = sRGB, 2 = linear; filter
# 0 = bilinear, 1 = bicubic, 2 = lanczos, for subsampled inputs only.
function(vqm_add_source_shaders prefix source half)
  foreach(chroma 0 1)
    foreach(transfer 0 1 2)
      if(chroma EQUAL 0)
        set(filters 0)
      else()
        set(filters 0 1 2)
      endif()
      foreach(filter IN LISTS filters)
        set(name ${prefix}_c${chroma}_t${transfer})
        if(chroma EQUAL 1)
          string(APPEND name _f${filter})
        endif()
        set(defs -DVQM_SRC_CHROMA=${chroma}
                 -DVQM_SRC_TRANSFER=${transfer}
                 -DVQM_SRC_FILTER=${filter})
        vqm_add_shader(${name} ${source} ${defs})
        if(half)
          vqm_add_shader(${name}_half ${source} ${defs}
                                 -DSS2_HALF_PLANES=1)
        endif()
      endforeach()
    endforeach()
  endforeach()
  # add_shader appends in this scope; pass both ordered lists back to the
  # caller so the embedding step sees every generated variant.
  set(VQM_SHADER_OUTPUTS "${VQM_SHADER_OUTPUTS}" PARENT_SCOPE)
  set(VQM_SHADER_BLOBS "${VQM_SHADER_BLOBS}" PARENT_SCOPE)
endfunction()

# Twelve prepare classes, each also built for 16-bit planes.
vqm_add_source_shaders(prepare metrics/ssimulacra2/shaders/prepare.comp TRUE)
vqm_add_shader(reduce metrics/ssimulacra2/shaders/reduce.comp)
foreach(variant IN LISTS VQM_SCALE_VARIANTS)
  separate_arguments(shape UNIX_COMMAND "${variant}")
  list(GET shape 0 rows)
  list(GET shape 1 cols)
  list(GET shape 2 lanes)
  set(defs -DSS2_SCALE_ROWS=${rows} -DSS2_SCALE_OUT_W=${cols}
           -DSS2_SCALE_LANES=${lanes})
  vqm_add_shader(scale_${rows}x${cols}x${lanes} metrics/ssimulacra2/shaders/scale.comp ${defs})
  vqm_add_shader(scale_${rows}x${cols}x${lanes}_half metrics/ssimulacra2/shaders/scale.comp ${defs}
    -DSS2_HALF_PLANES=1)
endforeach()

# Butteraugli (src/metrics/butteraugli/vulkan.cc) reuses the decode classes: its opsin
# shader is prepare.comp's class list, same order and membership (no 16-bit
# build -- the metric is f32 only), plus one blob per blur tap count and one
# per pass. The tap counts are libjxl's ComputeKernel radii for sigma 1.5642,
# 2.7, 3.2249 and 7.1559 (7, 13, 15 and 33 taps), and sigma 1.2's 5-tap blur
# lives inside butter_opsin.
vqm_add_source_shaders(butter_opsin metrics/butteraugli/shaders/butter_opsin.comp FALSE)
foreach(taps 7 13 15 33)
  vqm_add_shader(butter_blur_${taps} metrics/butteraugli/shaders/butter_blur.comp
                         -DBUTTER_BLUR_TAPS=${taps})
endforeach()
vqm_add_shader(butter_malta metrics/butteraugli/shaders/butter_malta.comp)
vqm_add_shader(butter_malta_lf metrics/butteraugli/shaders/butter_malta_lf.comp)
vqm_add_shader(butter_l2 metrics/butteraugli/shaders/butter_l2.comp)
vqm_add_shader(butter_erode metrics/butteraugli/shaders/butter_erode.comp)
vqm_add_shader(butter_reduce metrics/butteraugli/shaders/butter_reduce.comp)
vqm_add_shader(vmaf_luma metrics/vmaf/features/shaders/preprocess/vmaf_luma.comp)
vqm_add_source_shaders(vmaf_sdr metrics/vmaf/features/shaders/preprocess/vmaf_sdr.comp FALSE)
vqm_add_shader(vmaf_resize metrics/vmaf/features/shaders/preprocess/vmaf_resize.comp)
vqm_add_shader(vmaf_motion_horizontal metrics/vmaf/features/shaders/preprocess/vmaf_motion_horizontal.comp)
vqm_add_shader(vmaf_motion_reduce metrics/vmaf/features/shaders/preprocess/vmaf_motion_reduce.comp)
foreach(scale RANGE 0 3)
  vqm_add_shader(vmaf_vif_vertical_${scale} metrics/vmaf/features/shaders/vif/vmaf_vif_vertical.comp -DVIF_SCALE=${scale})
  vqm_add_shader(vmaf_vif_horizontal_${scale} metrics/vmaf/features/shaders/vif/vmaf_vif_horizontal.comp -DVIF_SCALE=${scale})
endforeach()
vqm_add_shader(vmaf_vif_fused0 metrics/vmaf/features/shaders/vif/vmaf_vif_fused0.comp -DVIF_SCALE=0)
vqm_add_shader(vmaf_vif_fused0_half metrics/vmaf/features/shaders/vif/vmaf_vif_fused0.comp -DVIF_SCALE=0 -DVQM_VIF_HALF_SOURCE=1)
foreach(scale RANGE 1 3)
  vqm_add_shader(vmaf_vif_fused_${scale} metrics/vmaf/features/shaders/vif/vmaf_vif_fused.comp -DVIF_SCALE=${scale})
  vqm_add_shader(vmaf_vif_down_fused_${scale} metrics/vmaf/features/shaders/vif/vmaf_vif_down_fused.comp -DVIF_SCALE=${scale})
endforeach()
vqm_add_shader(vmaf_vif_reduce metrics/vmaf/features/shaders/vif/vmaf_vif_reduce.comp)
vqm_add_shader(vmaf_adm_dwt metrics/vmaf/features/shaders/adm/vmaf_adm_dwt.comp)
vqm_add_shader(vmaf_adm_dwt_decouple metrics/vmaf/features/shaders/adm/vmaf_adm_dwt_decouple.comp)
vqm_add_shader(vmaf_adm_dwt_decouple_neg metrics/vmaf/features/shaders/adm/vmaf_adm_dwt_decouple.comp -DVQM_ADM_ENHN_GAIN_LIMIT=1)
vqm_add_shader(vmaf_adm_dwt_decouple_tiled metrics/vmaf/features/shaders/adm/vmaf_adm_dwt_decouple.comp -DVQM_ADM_SOURCE_TILE=1)
vqm_add_shader(vmaf_adm_dwt_decouple_tiled_neg metrics/vmaf/features/shaders/adm/vmaf_adm_dwt_decouple.comp -DVQM_ADM_SOURCE_TILE=1 -DVQM_ADM_ENHN_GAIN_LIMIT=1)
vqm_add_shader(vmaf_adm_accumulate metrics/vmaf/features/shaders/adm/vmaf_adm_accumulate.comp)
vqm_add_shader(vmaf_adm_reduce metrics/vmaf/features/shaders/adm/vmaf_adm_reduce.comp)
vqm_add_shader(vmaf_adm_finish metrics/vmaf/features/shaders/adm/vmaf_adm_finish.comp)
vqm_add_shader(vmaf_adm3_dwt_decouple metrics/vmaf/features/shaders/adm/vmaf_adm_dwt_decouple.comp -DVQM_ADM_ENHN_GAIN_LIMIT=1)
vqm_add_shader(vmaf_adm3_accumulate metrics/vmaf/features/shaders/adm/vmaf_adm3_accumulate.comp)
vqm_add_shader(vmaf_adm3_reduce metrics/vmaf/features/shaders/adm/vmaf_adm3_reduce.comp)
vqm_add_shader(vmaf_adm3_finish metrics/vmaf/features/shaders/adm/vmaf_adm3_finish.comp)
vqm_add_shader(vmaf_speed_filter_vertical metrics/vmaf/speed/shaders/vmaf_speed_filter_vertical.comp)
vqm_add_shader(vmaf_speed_filter_horizontal metrics/vmaf/speed/shaders/vmaf_speed_filter_horizontal.comp)
vqm_add_shader(vmaf_speed_stats_cov metrics/vmaf/speed/shaders/vmaf_speed_stats_cov.comp)
vqm_add_shader(vmaf_speed_stats_eigen metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp)
if(VQM_BUILD_TESTS AND BUILD_TESTING)
  # Same-device numerical oracle; excluded from library-only builds.
  vqm_add_shader(vmaf_speed_stats_eigen_legacy metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp
                 -DVQM_SPEED_FORCE_LEGACY_QR=1)
endif()
vqm_add_shader(vmaf_speed_stats_eigen_wide metrics/vmaf/speed/shaders/vmaf_speed_stats_eigen.comp -DVQM_SPEED_EIGEN_LANES=512)
vqm_add_shader(vmaf_speed_stats_pool metrics/vmaf/speed/shaders/vmaf_speed_stats_pool.comp)
vqm_add_shader(vmaf_speed_stats_reduce metrics/vmaf/speed/shaders/vmaf_speed_stats_reduce.comp)
vqm_add_shader(vmaf_speed_combine metrics/vmaf/speed/shaders/vmaf_speed_combine.comp)
vqm_add_shader(vmaf_cambi_prepare metrics/vmaf/cambi/shaders/vmaf_cambi_prepare.comp)
vqm_add_shader(vmaf_cambi_contrast metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp)
vqm_add_shader(vmaf_cambi_contrast_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_HALF_SOURCE=1)
vqm_add_shader(vmaf_cambi_contrast_subgroup metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_SUBGROUP_ROWS=1)
vqm_add_shader(vmaf_cambi_contrast_subgroup_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_SUBGROUP_ROWS=1 -DVQM_CAMBI_HALF_SOURCE=1)
vqm_add_shader(vmaf_cambi_decimate metrics/vmaf/cambi/shaders/vmaf_cambi_decimate.comp)
vqm_add_shader(vmaf_cambi_combine metrics/vmaf/cambi/shaders/vmaf_cambi_combine.comp)
vqm_add_shader(vmaf_cambi_pool_clear metrics/vmaf/cambi/shaders/pool/vmaf_cambi_pool_clear.comp)
vqm_add_shader(vmaf_cambi_pool_histogram metrics/vmaf/cambi/shaders/pool/vmaf_cambi_pool_histogram.comp)
vqm_add_shader(vmaf_cambi_pool_select metrics/vmaf/cambi/shaders/pool/vmaf_cambi_pool_select.comp)
vqm_add_shader(vmaf_cambi_pool_partial metrics/vmaf/cambi/shaders/pool/vmaf_cambi_pool_partial.comp)
vqm_add_shader(vmaf_cambi_pool_finish metrics/vmaf/cambi/shaders/pool/vmaf_cambi_pool_finish.comp)

# ColorVideoVDP: metric-local physical photometry, temporal FIR and spatial model.
foreach(filter RANGE 0 2)
  vqm_add_shader(cvvdp_prepare_${filter} metrics/cvvdp/shaders/cvvdp_prepare.comp
    -DVQM_SRC_CHROMA=1 -DVQM_SRC_TRANSFER=0 -DVQM_SRC_FILTER=${filter})
endforeach()
foreach(stage temporal pyramid contrast mask reduce)
  vqm_add_shader(cvvdp_${stage} metrics/cvvdp/shaders/cvvdp_${stage}.comp)
endforeach()

# PSNR-HVS: raw coded planes, FP32 block transforms/masking and reduction.
foreach(stage prepare blocks reduce)
  vqm_add_shader(psnr_hvs_${stage} metrics/psnr_hvs/shaders/psnr_hvs_${stage}.comp)
endforeach()

string(REPLACE ";" "," VQM_SHADER_NAMES "${VQM_SHADER_BLOBS}")
string(REPLACE ";" "\n" VQM_SHADER_MANIFEST "${VQM_SHADER_BLOBS}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_shader_manifest.txt"
  CONTENT "${VQM_SHADER_MANIFEST}\n")
set(VQM_VULKAN_SHADER_HEADER
    "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_shaders.h")
# copy_if_different leaves the header timestamp unchanged when shader bytes
# match. The stamp keeps Make from re-running the embedder on every build.
set(VQM_SHADER_STAMP "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan_shaders.stamp")
add_custom_command(
  OUTPUT "${VQM_SHADER_STAMP}" "${VQM_VULKAN_SHADER_HEADER}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
  COMMAND ${CMAKE_COMMAND}
          "-DSHADER_DIR=${CMAKE_CURRENT_BINARY_DIR}/shaders"
          "-DOUTPUT=${VQM_VULKAN_SHADER_HEADER}"
          "-DSHADERS=${VQM_SHADER_NAMES}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
  COMMAND ${CMAKE_COMMAND} -E touch "${VQM_SHADER_STAMP}"
  DEPENDS ${VQM_SHADER_OUTPUTS} cmake/embed_spirv.cmake
  COMMENT "embedding SPIR-V shaders"
  VERBATIM)
# One producer owns the generated header shared by the timed and untimed cores.
# Attaching the same custom-command output to both targets lets Make run the
# embedding step concurrently during an incremental parallel build.
add_custom_target(vqm_shaders DEPENDS "${VQM_SHADER_STAMP}" "${VQM_VULKAN_SHADER_HEADER}")

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
  src/vulkan/vulkan_runtime_inputs.cc
  src/vulkan/vulkan_shared_storage.cc)

# Internal feature headers are shared by the core and its tests, not exported.
set(VQM_INTERNAL_INCLUDE_DIRS src)

# The core metrics take decoded pixel planes and link only the Vulkan
# loader.
function(vqm_configure_core target)
  target_sources(${target} PRIVATE ${VQM_CORE_SOURCES})
  add_dependencies(${target} vqm_shaders)
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
                include/vqm_vulkan.h include/vqm_pool.h include/vqm_memory.h include/vqm_memory_types.h include/vqm_shared.h include/vqm_metrics.h
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

# Benchmark tools: a second core build with opt-in stage timers, plus the
# driver. Needs image decoding, so it follows the CLI switch by default.
if(VQM_BUILD_BENCH)
  add_library(vqm_core_timing STATIC)
  vqm_configure_core(vqm_core_timing)
  target_compile_definitions(vqm_core_timing
    PRIVATE VQM_STAGE_TIMING)

  add_executable(vqm_bench
    bench/driver/main.cc
    bench/driver/ssimulacra2.cc
    bench/driver/butteraugli.cc
    bench/driver/psnr_hvs.cc
    bench/driver/common.cc
    bench/driver/batch_modes.cc
    bench/driver/options.cc
    bench/driver/metric_pair.cc
    bench/driver/vmaf.cc
    bench/driver/cvvdp.cc
    bench/driver/workers.cc)
  target_include_directories(vqm_bench PRIVATE ${VQM_INTERNAL_INCLUDE_DIRS} tools/image_io tools bench)
  target_compile_features(vqm_bench PRIVATE cxx_std_20)
  target_compile_definitions(vqm_bench PRIVATE VQM_STAGE_TIMING)
  target_link_libraries(vqm_bench
    PRIVATE vqm_core_timing vqm_io vqm_warnings)
endif()

if(VQM_BUILD_TESTS AND BUILD_TESTING)
  add_executable(vqm_test
    tests/support.cc
    tests/metrics/psnr_hvs/pipeline_test.cc
    tests/metrics/cvvdp/cvvdp_test.cc
    tests/metrics/cvvdp/interop_test.cc
    tests/metrics/cvvdp/memory_test.cc
    tests/metrics/cvvdp/spatial_test.cc
    tests/metrics/cvvdp/temporal_test.cc
    tests/core/runtime_test.cc
    tests/metrics/ssimulacra2/pipeline_test.cc
    tests/metrics/ssimulacra2/score_test.cc
    tests/core/memory_test.cc
    tests/core/memory_plan_test.cc
    tests/core/shared_metrics_test.cc
    tests/core/metrics_test.cc
    tests/core/alpha_test.cc
    tests/core/pool_test.cc
    tests/core/image_view_test.cc
    tests/core/vulkan_interop_test.cc
    tests/input/input_format_test.cc
    tests/input/input_format_chroma_test.cc
    tests/metrics/butteraugli/butter_ref.cc
    tests/metrics/butteraugli/butter_test.cc
    tests/metrics/vmaf/features/vmaf_luma_test.cc
    tests/metrics/vmaf/features/vmaf_resize_test.cc
    tests/metrics/vmaf/features/vmaf_adm_test.cc
    tests/metrics/vmaf/features/vmaf_adm3_test.cc
    tests/metrics/vmaf/cambi/vmaf_cambi_prepare_test.cc
    tests/metrics/vmaf/cambi/vmaf_cambi_contrast_test.cc
    tests/metrics/vmaf/cambi/vmaf_cambi_decimate_test.cc
    tests/metrics/vmaf/cambi/vmaf_cambi_test.cc
    tests/metrics/vmaf/cambi/vmaf_cambi_pool_test.cc
    tests/metrics/vmaf/features/vmaf_feature_pipeline_test.cc
    tests/metrics/vmaf/model/vmaf_model_test.cc
    tests/metrics/vmaf/features/vmaf_motion_test.cc
    tests/metrics/vmaf/pipeline/vmaf_pipeline_test.cc
    tests/metrics/vmaf/pipeline/vmaf_memory_test.cc
    tests/metrics/vmaf/pipeline/vmaf_sequence_test.cc
    tests/metrics/vmaf/pipeline/vmaf_temporal_test.cc
    tests/metrics/vmaf/speed/vmaf_speed_filter_test.cc
    tests/metrics/vmaf/speed/vmaf_speed_test.cc
    tests/metrics/vmaf/speed/vmaf_speed_stats_test.cc
    tests/metrics/vmaf/model/vmaf_v1_model_test.cc
    tests/metrics/vmaf/pipeline/vmaf_v1_motion_test.cc
    tests/metrics/vmaf/pipeline/vmaf_v1_sequence_test.cc
    tests/metrics/vmaf/pipeline/vmaf_v1_temporal_test.cc
    tests/metrics/vmaf/pipeline/vmaf_v1_pipeline_test.cc
    tests/metrics/vmaf/pipeline/vmaf_v1_external_test.cc
    tests/metrics/vmaf/features/vmaf_vif_test.cc)
  target_include_directories(vqm_test PRIVATE ${VQM_INTERNAL_INCLUDE_DIRS} tools/image_io tests)
  target_compile_features(vqm_test PRIVATE cxx_std_20)
  # The input tests load real files, so the decoders are linked in too.
  target_link_libraries(vqm_test
    PRIVATE vqm_core vqm_io vqm_warnings Vulkan::Vulkan)
  add_test(NAME core COMMAND vqm_test)
  add_executable(vqm_sequence_io_test tests/input/sequence_reader_test.cc)
  target_include_directories(vqm_sequence_io_test PRIVATE include tools/image_io)
  target_compile_features(vqm_sequence_io_test PRIVATE cxx_std_20)
  target_link_libraries(vqm_sequence_io_test PRIVATE vqm_io vqm_core vqm_warnings)
  add_test(NAME sequence_io COMMAND vqm_sequence_io_test)


  if(VQM_INSTALL)
    include(cmake/consumer_settings.cmake)
    set(VQM_CONSUMER_SETTINGS "${CMAKE_CURRENT_BINARY_DIR}/package_consumer_settings.cmake")
    vqm_write_consumer_settings("${VQM_CONSUMER_SETTINGS}")
    add_test(NAME package_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DVQM_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        "-DVQM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DVQM_CONSUMER_SETTINGS=${VQM_CONSUMER_SETTINGS}"
        "-DVQM_CONFIG=$<CONFIG>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer.cmake")
  endif()

  add_executable(vqm_bench_helpers_test tests/core/bench_helpers_test.cc)
  target_include_directories(vqm_bench_helpers_test PRIVATE bench)
  target_compile_features(vqm_bench_helpers_test PRIVATE cxx_std_20)
  target_link_libraries(vqm_bench_helpers_test
    PRIVATE Threads::Threads vqm_warnings)
  add_test(NAME benchmark_helpers COMMAND vqm_bench_helpers_test)

  find_package(Python3 3.11 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME cid22_fetch
      COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests
              -p test_fetch_cid22.py -v)
    set_tests_properties(cid22_fetch PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME reference_fetch
      COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests
              -p test_reference_fetch.py -v)
    set_tests_properties(reference_fetch PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME vmaf_oracle
      COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests
              -p test_vmaf_oracle.py -v)
    set_tests_properties(vmaf_oracle PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME vmaf_cid22
      COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests
              -p test_vmaf_cid22.py -v)
    set_tests_properties(vmaf_cid22 PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    if(VQM_BUILD_CLI)
      add_test(NAME cli_integration
        COMMAND "${CMAKE_COMMAND}" -E env
                "VQM_CLI_BINARY=$<TARGET_FILE:vqm>"
                "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>"
                "${Python3_EXECUTABLE}" -m unittest discover -s tests
                -p test_cli.py -v)
      add_test(NAME video_cli
        COMMAND "${CMAKE_COMMAND}" -E env
                "VQM_CLI_BINARY=$<TARGET_FILE:vqm>"
                "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>"
                "${Python3_EXECUTABLE}" -m unittest discover -s tests
                -p test_video_cli.py -v)
      if(TARGET vqm_cli_nvdec)
        add_test(NAME nvdec_cli
          COMMAND "${CMAKE_COMMAND}" -E env
                  "VQM_CLI_BINARY=$<TARGET_FILE:vqm>"
                  "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>"
                  "${Python3_EXECUTABLE}" -m unittest discover -s tests
                  -p test_nvdec_cli.py -v)
        set_tests_properties(nvdec_cli PROPERTIES
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
      endif()
      set(VQM_CVVDP_BENCH_ENV "--unset=VQM_BENCH_BINARY")
      if(VQM_BUILD_BENCH)
        set(VQM_CVVDP_BENCH_ENV "VQM_BENCH_BINARY=$<TARGET_FILE:vqm_bench>")
      endif()
      add_test(NAME cvvdp_tools
        COMMAND "${CMAKE_COMMAND}" -E env
          "VQM_CLI_BINARY=$<TARGET_FILE:vqm>"
          "${VQM_CVVDP_BENCH_ENV}"
          "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>"
          "${Python3_EXECUTABLE}" -m unittest discover -s tests -p test_cvvdp_tools.py -v)
      set_tests_properties(cli_integration video_cli cvvdp_tools PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    add_test(NAME profile_scripts
      COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests
              -p test_profile_scripts.py -v)
    set_tests_properties(profile_scripts PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
      RESULT_VARIABLE VQM_NUMPY_STATUS
      OUTPUT_QUIET ERROR_QUIET)
    if(VQM_NUMPY_STATUS EQUAL 0)
      foreach(test_name agreement agreement_adapters agreement_audit all_metric_correlation
          gate_defaults spatial_reference vmaf_agreement temporal_fetch build_references
          video_benchmark cli_results)
        add_test(NAME ${test_name}
          COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests
            -p test_${test_name}.py -v)
        set_tests_properties(${test_name} PROPERTIES
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
      endforeach()
      if(VQM_BUILD_CLI)
        add_test(NAME temporal_tools
          COMMAND "${CMAKE_COMMAND}" -E env
            "VQM_TEMPORAL_BINARY=$<TARGET_FILE:vqm>"
            "VQM_CLI_BINARY=$<TARGET_FILE:vqm>"
            "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>"
            "${Python3_EXECUTABLE}" -m unittest discover -s tests
            -p test_temporal_tools.py -v)
        set_tests_properties(temporal_tools PROPERTIES
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
      endif()
      foreach(test_name benchmark_tools validation_tools)
        if(test_name STREQUAL benchmark_tools)
          set(test_file test_bench_tools.py)
        else()
          set(test_file test_validation.py)
        endif()
        add_test(NAME ${test_name}
          COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tests
                  -p ${test_file} -v)
        set_tests_properties(${test_name} PROPERTIES
          WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
      endforeach()
    else()
      message(STATUS "NumPy not found; Python benchmark/validation tests skipped")
    endif()
    if(VQM_BUILD_BENCH)
      add_test(NAME benchmark_integration
        COMMAND "${CMAKE_COMMAND}" -E env
                "VQM_BENCH_BINARY=$<TARGET_FILE:vqm_bench>"
                "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>"
                "${Python3_EXECUTABLE}" -m unittest discover -s tests
                -p test_bench_integration.py -v)
      set_tests_properties(benchmark_integration PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
  endif()
endif()
