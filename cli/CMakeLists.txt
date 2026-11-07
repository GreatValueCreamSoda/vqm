add_executable(vqm main.cc video.cc)
set_target_properties(vqm PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
target_include_directories(vqm PRIVATE "${PROJECT_SOURCE_DIR}/src"
  "${PROJECT_SOURCE_DIR}/tools/image_io")
target_compile_features(vqm PRIVATE cxx_std_20)
target_link_libraries(vqm PRIVATE vqm_core vqm_io vqm_warnings)

if(VQM_ENABLE_BESTSOURCE)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "VQM_ENABLE_BESTSOURCE requires Linux Vulkan opaque FD interop")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(VQM_BESTSOURCE REQUIRED IMPORTED_TARGET
    bestsource>=23 libavutil>=61.6.0)
  find_path(VQM_BESTSOURCE_GPU_INCLUDE_DIR gpuhash.h
    DOC "BestSource src directory containing the GPU export header")
  if(NOT EXISTS "${VQM_BESTSOURCE_GPU_INCLUDE_DIR}/gpuhash.h")
    message(FATAL_ERROR "Set VQM_BESTSOURCE_GPU_INCLUDE_DIR to the GPU-enabled BestSource src directory")
  endif()
  target_sources(vqm PRIVATE bestsource.cc bestsource_interop.cc)
  target_include_directories(vqm SYSTEM PRIVATE "${VQM_BESTSOURCE_GPU_INCLUDE_DIR}")
  target_compile_definitions(vqm PRIVATE VQM_CLI_BESTSOURCE BS_GPU=1)
  target_link_libraries(vqm PRIVATE PkgConfig::VQM_BESTSOURCE)
endif()

if(VQM_INSTALL)
  install(TARGETS vqm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
