# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.

# Local C++ and Python test registration.

add_executable(vqm_test
  support.cc
  metrics/psnr_hvs/pipeline_test.cc
  metrics/cvvdp/cvvdp_test.cc
  metrics/cvvdp/interop_test.cc
  metrics/cvvdp/memory_test.cc
  metrics/cvvdp/spatial_test.cc
  metrics/cvvdp/temporal_test.cc
  core/runtime_test.cc
  metrics/ssimulacra2/pipeline_test.cc
  metrics/ssimulacra2/score_test.cc
  core/memory_test.cc
  core/memory_plan_test.cc
  core/shared_metrics_test.cc
  core/metric_scratch_test.cc
  core/metrics_test.cc
  core/alpha_test.cc
  core/pool_test.cc
  core/image_view_test.cc
  core/vulkan_interop_test.cc
  input/input_format_test.cc
  input/input_format_chroma_test.cc
  metrics/butteraugli/butter_ref.cc
  metrics/butteraugli/butter_test.cc
  metrics/vmaf/features/vmaf_luma_test.cc
  metrics/vmaf/features/vmaf_resize_test.cc
  metrics/vmaf/features/vmaf_adm_test.cc
  metrics/vmaf/features/vmaf_adm3_test.cc
  metrics/vmaf/cambi/vmaf_cambi_prepare_test.cc
  metrics/vmaf/cambi/vmaf_cambi_contrast_test.cc
  metrics/vmaf/cambi/vmaf_cambi_decimate_test.cc
  metrics/vmaf/cambi/vmaf_cambi_test.cc
  metrics/vmaf/cambi/vmaf_cambi_pool_test.cc
  metrics/vmaf/features/vmaf_feature_pipeline_test.cc
  metrics/vmaf/model/vmaf_model_test.cc
  metrics/vmaf/features/vmaf_motion_test.cc
  metrics/vmaf/pipeline/vmaf_pipeline_test.cc
  metrics/vmaf/pipeline/vmaf_memory_test.cc
  metrics/vmaf/pipeline/vmaf_sequence_test.cc
  metrics/vmaf/pipeline/vmaf_temporal_test.cc
  metrics/vmaf/speed/vmaf_speed_filter_test.cc
  metrics/vmaf/speed/vmaf_speed_test.cc
  metrics/vmaf/speed/vmaf_speed_stats_test.cc
  metrics/vmaf/model/vmaf_v1_model_test.cc
  metrics/vmaf/pipeline/vmaf_v1_motion_test.cc
  metrics/vmaf/pipeline/vmaf_v1_sequence_test.cc
  metrics/vmaf/pipeline/vmaf_v1_temporal_test.cc
  metrics/vmaf/pipeline/vmaf_v1_pipeline_test.cc
  metrics/vmaf/pipeline/vmaf_v1_external_test.cc
  metrics/vmaf/features/vmaf_vif_test.cc)
target_include_directories(vqm_test PRIVATE ${VQM_INTERNAL_INCLUDE_DIRS}
  "${PROJECT_SOURCE_DIR}/tools/image_io" "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(vqm_test PRIVATE cxx_std_20)
# The input tests load real files, so the decoders are linked in too.
target_link_libraries(vqm_test
  PRIVATE vqm_core vqm_io vqm_warnings Vulkan::Vulkan)
add_test(NAME core COMMAND vqm_test)
add_executable(vqm_sequence_io_test input/sequence_reader_test.cc)
target_include_directories(vqm_sequence_io_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/tools/image_io")
target_compile_features(vqm_sequence_io_test PRIVATE cxx_std_20)
target_link_libraries(vqm_sequence_io_test PRIVATE vqm_io vqm_core vqm_warnings)
add_test(NAME sequence_io COMMAND vqm_sequence_io_test)


if(VQM_INSTALL)
  include("${PROJECT_SOURCE_DIR}/cmake/consumer_settings.cmake")
  set(VQM_CONSUMER_SETTINGS "${PROJECT_BINARY_DIR}/package_consumer_settings.cmake")
  vqm_write_consumer_settings("${VQM_CONSUMER_SETTINGS}")
  add_test(NAME package_consumer
    COMMAND "${CMAKE_COMMAND}"
      "-DVQM_BUILD_DIR=${PROJECT_BINARY_DIR}"
      "-DVQM_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
      "-DVQM_CONSUMER_SETTINGS=${VQM_CONSUMER_SETTINGS}"
      "-DVQM_CONFIG=$<CONFIG>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/package_consumer.cmake")
endif()

add_executable(vqm_bench_helpers_test core/bench_helpers_test.cc)
target_include_directories(vqm_bench_helpers_test PRIVATE "${PROJECT_SOURCE_DIR}/bench")
target_compile_features(vqm_bench_helpers_test PRIVATE cxx_std_20)
target_link_libraries(vqm_bench_helpers_test
  PRIVATE Threads::Threads vqm_warnings)
add_test(NAME benchmark_helpers COMMAND vqm_bench_helpers_test)

set_target_properties(vqm_test vqm_sequence_io_test vqm_bench_helpers_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set_tests_properties(core sequence_io benchmark_helpers PROPERTIES
  WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

# Keep discovery, environments and source working directories consistent.
function(vqm_add_python_test name file)
  cmake_parse_arguments(TEST "" "" "ENV" ${ARGN})
  add_test(NAME ${name}
    COMMAND "${CMAKE_COMMAND}" -E env ${TEST_ENV}
      "${Python3_EXECUTABLE}" -m unittest discover -s tests -p ${file} -v)
  set_tests_properties(${name} PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endfunction()

find_package(Python3 3.11 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
  return()
endif()

vqm_add_python_test(cid22_fetch test_fetch_cid22.py)
vqm_add_python_test(reference_fetch test_reference_fetch.py)
vqm_add_python_test(vmaf_oracle test_vmaf_oracle.py)
vqm_add_python_test(vmaf_cid22 test_vmaf_cid22.py)
if(VQM_BUILD_CLI)
  set(cli_environment "VQM_CLI_BINARY=$<TARGET_FILE:vqm>"
                      "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>")
  vqm_add_python_test(cli_integration test_cli.py ENV ${cli_environment})
  vqm_add_python_test(video_cli test_video_cli.py ENV ${cli_environment})
  if(VQM_ENABLE_BESTSOURCE)
    vqm_add_python_test(bestsource_cli test_bestsource_cli.py ENV ${cli_environment})
  endif()
  set(bench_environment "--unset=VQM_BENCH_BINARY")
  if(VQM_BUILD_BENCH)
    set(bench_environment "VQM_BENCH_BINARY=$<TARGET_FILE:vqm_bench>")
  endif()
  vqm_add_python_test(cvvdp_tools test_cvvdp_tools.py
    ENV ${bench_environment} ${cli_environment})
endif()
vqm_add_python_test(profile_scripts test_profile_scripts.py)
vqm_add_python_test(throughput_comparison test_throughput_comparison.py)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
  RESULT_VARIABLE VQM_NUMPY_STATUS OUTPUT_QUIET ERROR_QUIET)
if(VQM_NUMPY_STATUS EQUAL 0)
  foreach(test_name agreement agreement_adapters agreement_audit all_metric_correlation
      gate_defaults build_paths spatial_reference vmaf_agreement temporal_fetch build_references
      video_benchmark cli_results temporal_tools)
    vqm_add_python_test(${test_name} test_${test_name}.py)
  endforeach()
  vqm_add_python_test(benchmark_tools test_bench_tools.py)
  vqm_add_python_test(validation_tools test_validation.py)
else()
  message(STATUS "NumPy not found; Python benchmark/validation tests skipped")
endif()
if(VQM_BUILD_BENCH)
  vqm_add_python_test(benchmark_integration test_bench_integration.py
    ENV "VQM_BENCH_BINARY=$<TARGET_FILE:vqm_bench>"
        "VQM_TEST_BINARY=$<TARGET_FILE:vqm_test>")
endif()
