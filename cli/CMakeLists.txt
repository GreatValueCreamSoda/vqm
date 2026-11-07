string(TOUPPER "${VQM_ENABLE_NVDEC}" nvdec_mode)
if(NOT nvdec_mode MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "VQM_ENABLE_NVDEC must be AUTO, ON or OFF")
endif()
if(NOT nvdec_mode STREQUAL "OFF" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(CUDAToolkit 11.8 QUIET)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(VQM_FFMPEG QUIET IMPORTED_TARGET
      libavformat>=60 libavcodec>=60 libavutil>=58.29.100)
  endif()
endif()

if(NOT nvdec_mode STREQUAL "OFF" AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
    AND CUDAToolkit_FOUND AND CUDAToolkit_NVCC_EXECUTABLE AND TARGET CUDA::cuda_driver
    AND TARGET PkgConfig::VQM_FFMPEG)
  set(pack_ptx "${CMAKE_CURRENT_BINARY_DIR}/nvdec_pack.ptx")
  set(pack_header "${CMAKE_CURRENT_BINARY_DIR}/nvdec_pack_ptx.h")
  add_custom_command(
    OUTPUT "${pack_header}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --ptx --gpu-architecture=compute_75
            "${CMAKE_CURRENT_SOURCE_DIR}/nvdec_pack.cu" -o "${pack_ptx}"
    COMMAND "${CMAKE_COMMAND}" "-DPTX=${pack_ptx}" "-DHEADER=${pack_header}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_ptx.cmake"
    DEPENDS nvdec_pack.cu embed_ptx.cmake
    BYPRODUCTS "${pack_ptx}"
    VERBATIM)
  add_library(vqm_cli_nvdec STATIC nvdec.cc nvdec_interop.cc "${pack_header}")
  target_compile_features(vqm_cli_nvdec PRIVATE cxx_std_20)
  target_include_directories(vqm_cli_nvdec PRIVATE "${CMAKE_CURRENT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/tools/image_io")
  target_link_libraries(vqm_cli_nvdec PRIVATE vqm_core vqm_warnings
    CUDA::cuda_driver PkgConfig::VQM_FFMPEG)
  message(STATUS "vqm CLI NVDEC enabled")
elseif(nvdec_mode STREQUAL "ON")
  message(FATAL_ERROR "VQM_ENABLE_NVDEC=ON requires Linux, CUDA Toolkit 11.8+ with nvcc and FFmpeg 6.1+ development libraries")
elseif(nvdec_mode STREQUAL "AUTO")
  message(STATUS "vqm CLI NVDEC unavailable (requires Linux, CUDA Toolkit 11.8+ and FFmpeg 6.1+ development libraries)")
endif()

add_executable(vqm main.cc video.cc)
set_target_properties(vqm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_include_directories(vqm PRIVATE "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tools/image_io")
target_compile_features(vqm PRIVATE cxx_std_20)
target_link_libraries(vqm PRIVATE vqm_core vqm_io vqm_warnings)
if(TARGET vqm_cli_nvdec)
  target_compile_definitions(vqm PRIVATE VQM_CLI_NVDEC)
  target_link_libraries(vqm PRIVATE vqm_cli_nvdec)
endif()

if(VQM_INSTALL)
  install(TARGETS vqm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
