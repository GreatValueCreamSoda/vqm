# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.

# Metric shader variants and the single SPIR-V embedding producer.

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
vqm_add_shader(vmaf_hdr metrics/vmaf/features/shaders/preprocess/vmaf_hdr.comp)
vqm_add_shader(vmaf_hdr_resize metrics/vmaf/features/shaders/preprocess/vmaf_hdr_resize.comp)
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
vqm_add_shader(vmaf_speed_prescale metrics/vmaf/speed/shaders/vmaf_speed_prescale.comp)
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
vqm_add_shader(vmaf_cambi_contrast_pq metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_PQ=1)
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
