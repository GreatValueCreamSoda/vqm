if(NOT VQM_BUILD_DIR OR NOT VQM_SOURCE_DIR OR NOT VQM_CONSUMER_SETTINGS)
  message(FATAL_ERROR "package consumer test needs source, build and toolchain settings")
endif()
include("${VQM_CONSUMER_SETTINGS}")
set(configure_options -C "${VQM_CONSUMER_SETTINGS}" -G "${VQM_CONSUMER_GENERATOR}")
if(VQM_CONSUMER_GENERATOR_PLATFORM)
  list(APPEND configure_options -A "${VQM_CONSUMER_GENERATOR_PLATFORM}")
endif()
if(VQM_CONSUMER_GENERATOR_TOOLSET)
  list(APPEND configure_options -T "${VQM_CONSUMER_GENERATOR_TOOLSET}")
endif()
set(test_dir "${VQM_BUILD_DIR}/package-consumer-test")
file(REMOVE_RECURSE "${test_dir}")
file(MAKE_DIRECTORY "${test_dir}/sdk-include")
file(WRITE "${test_dir}/sdk-include/vqm_sdk_probe.h" "#pragma once\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${VQM_BUILD_DIR}"
          --prefix "${test_dir}/prefix" --config "${VQM_CONFIG}"
  RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "package installation failed: ${status}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${VQM_SOURCE_DIR}/tests/package_consumer"
          -B "${test_dir}/build" ${configure_options}
          "-DVQM_TEST_PREFIX=${test_dir}/prefix"
          "-DCMAKE_BUILD_TYPE=${VQM_CONFIG}"
          "-DVQM_TEST_SDK_DIR=${test_dir}/sdk-include"
  RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "external consumer configuration failed: ${status}")
endif()
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${test_dir}/build" --config "${VQM_CONFIG}"
  RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "external consumer build failed: ${status}")
endif()
if(NOT VQM_CONSUMER_CROSSCOMPILING)
  execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${test_dir}/build"
            --build-config "${VQM_CONFIG}" --output-on-failure
    RESULT_VARIABLE status)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "external consumer failed: ${status}")
  endif()
else()
  message(STATUS "Cross-compiling: consumer built; execution skipped")
endif()

# Configuration alone verifies add_subdirectory defaults without compiling a
# second copy of all embedded shaders.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${VQM_SOURCE_DIR}/tests/embedded_consumer"
          -B "${test_dir}/embedded" ${configure_options}
          "-DVQM_SOURCE_DIR=${VQM_SOURCE_DIR}"
  RESULT_VARIABLE status)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "embedded consumer configuration failed: ${status}")
endif()
