# FP16 operands with FP32 cooperative-matrix accumulation

On 2026-11-05, a standalone correctness probe enabled and executed
`VK_KHR_cooperative_matrix` on the local NVIDIA GeForce RTX 5070, driver
615.78.08. Unlike the previous scalar half-product experiments, this path
retains information in FP16 products before the FP32 result. It does not
restore operand bits discarded by conversion to FP16. No throughput claim
comes from this probe; it has no timers.

## API and compiler contract

The supported type/shape combinations and shader stages must be queried,
rather than inferred from extension presence. This device reports subgroup
size 32 and compute-stage FP16 A/B with FP32 C/result shapes 16×16×16,
16×8×16 and 16×8×8. All three were executed. The
[Khronos cooperative-matrix specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html)
defines the extension and query; its
[type/shape matching rules](https://docs.vulkan.org/spec/latest/chapters/shaders.html#shaders-cooperative-matrices)
apply to each operation.

The probe creates its own Vulkan 1.1 instance/device. Its enabled extensions
and features are:

| Extension | Enabled feature |
| --- | --- |
| `VK_KHR_cooperative_matrix` | `VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix` |
| `VK_KHR_shader_float16_int8` | `VkPhysicalDeviceShaderFloat16Int8Features::shaderFloat16` |
| `VK_KHR_vulkan_memory_model` | `VkPhysicalDeviceVulkanMemoryModelFeatures::vulkanMemoryModel` |
| `VK_EXT_subgroup_size_control` | `VkPhysicalDeviceSubgroupSizeControlFeatures::computeFullSubgroups` |

Other queried subfeatures are disabled. In particular, it does not request
robust matrix accesses or 16-bit storage buffers: FP32 SSBO inputs are converted
into shared `float16_t` arrays. The
[cooperative feature structure](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceCooperativeMatrixFeaturesKHR.html)
must be supplied both when querying support and when enabling the feature.

Working GLSL uses `GL_KHR_cooperative_matrix`,
`GL_EXT_shader_explicit_arithmetic_types_float16`,
`GL_KHR_memory_scope_semantics` and `#pragma use_vulkan_memory_model`.
`coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>` and the matching B
operand feed a `coopmat<float, ..., gl_MatrixUseAccumulator>` using
`coopMatLoad`, `coopMatMulAdd` and `coopMatStore`. Scope-wide control flow and
arguments are uniform. The
[GLSL extension](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GLSL_KHR_cooperative_matrix.txt)
defines those operations and their invocation requirements.

`glslc --target-env=vulkan1.1 --target-spv=spv1.3 -O` successfully compiles the
shader. All three modules pass `spirv-val --target-env vulkan1.1` and contain
`Float16`, `VulkanMemoryModel`, `CooperativeMatrixKHR`, `OpMemoryModel Logical
Vulkan` and `OpCooperativeMatrixMulAddKHR`. SPIR-V proves use of the matrix
operation; it does not identify the physical instruction or measure its speed.

The pipeline sets
`VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`. This is required
for these cooperative-matrix pipelines with SPIR-V earlier than 1.6; merely
launching 32 invocations is insufficient to satisfy the current API contract.
The same
[Vulkan SPIR-V appendix](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
also states that matrix multiply-add evaluation order and internal precision are
implementation-dependent. FP32 C/result types therefore do not provide a
portable guarantee of sequential IEEE FP32 accumulation or a fully preserved
product. The values below are measurements of this GPU/driver.

## Numerical evidence

Each shape ran six scenes, for 18 cases. The final run passed with Khronos
validation and synchronization validation enabled and no reported errors.
For 16×16×16, representative output elements were:

| Scene | Original-input exact sum | FP16-operand exact sum | GPU result | Sum after rounding each product to FP16 |
| --- | ---: | ---: | ---: | ---: |
| `100.5²`, 16 terms | 161604 | 161604 | 161604 | 161664 |
| Alternating `100.5²` and `-100.5×100.25` | 201 | 201 | 201 | 256 |
| `512²`, 16 terms | 4194304 | 4194304 | 4194304 | Nonfinite |
| Small residual from `(1 + 2⁻¹⁰)² − 2(1 + 2⁻¹⁰) + 1` | 0.000003814697265625 | Same | Same | 0 |
| Alternating `−127` and `127 + 1/256` | 0.03125 | 0 | 0 | 0 |
| Deterministic mixed-sign row/column | 64.1368887424469 | Same | 64.13688659667969 | 64.113037109375 |

The small-residual expression repeats four times in the 16-term dot product.
The five structured scenes agreed exactly with their FP32-representable
FP16-operand references for every output element. The mixed-sign case had
maximum error 7.450580596923828e-6 for K=16 and 6.67572021484375e-6 for K=8
against the double sum of FP16 operands. These are observed errors on finite
examples, not worst-case bounds over valid VMAF inputs.

This confirms the scalar half-product counterexample does not disqualify a
wide matrix path when both operands are exact FP16 values. Operand
quantization, conditioning, reordering, packing cost and limited matrix-unit
utilization remain separate questions. NVIDIA's
[Tensor Core background guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
and [programming introduction](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
describe full-product FP16 input with FP32 accumulation on NVIDIA hardware;
the Vulkan precision contract remains the portable boundary.

## Reproduction and limits

The maintained probe and runner are under
[`../../../experiments/cooperative_matrix/`](../../../experiments/cooperative_matrix/README.md).
Run `python3 bench/experiments/cooperative_matrix/run_platform.py --validation`
from the repository root. Current logs, full per-case JSON, source/module/binary
hashes and commands are under
`artifacts/reports/vmaf-cooperative-matrix/platform/`. The executable creates a
private device; the production context and public API are unchanged.

This is a correctness/capability experiment. It does not measure VMAF stage
speed, classify all conditioning failures, establish any other vendor's
precision, or qualify enabling a production default. A VMAF backend must keep
feature fallbacks, input precision and sequence history contracts explicit and
pass the independent metric gates before adoption.

The [compact platform snapshot](../../measurements/vmaf-cooperative-platform.json)
preserves the 18 cases independently of local build/report directories.
