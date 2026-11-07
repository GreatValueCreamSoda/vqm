# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.

# Timing-enabled core and benchmark driver.

add_library(vqm_core_timing STATIC)
vqm_configure_core(vqm_core_timing)
set_target_properties(vqm_core_timing PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_compile_definitions(vqm_core_timing
  PRIVATE VQM_STAGE_TIMING)

add_executable(vqm_bench
  driver/main.cc
  driver/ssimulacra2.cc
  driver/butteraugli.cc
  driver/psnr_hvs.cc
  driver/common.cc
  driver/batch_modes.cc
  driver/options.cc
  driver/metric_pair.cc
  driver/vmaf.cc
  driver/cvvdp.cc
  driver/workers.cc)
set_target_properties(vqm_bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_include_directories(vqm_bench PRIVATE ${VQM_INTERNAL_INCLUDE_DIRS}
  "${PROJECT_SOURCE_DIR}/tools/image_io" "${PROJECT_SOURCE_DIR}/tools"
  "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(vqm_bench PRIVATE cxx_std_20)
target_compile_definitions(vqm_bench PRIVATE VQM_STAGE_TIMING)
target_link_libraries(vqm_bench
  PRIVATE vqm_core_timing vqm_io vqm_warnings)
