# C++ library

Install the core and its CMake package into a prefix.

```sh
cmake --install build --prefix /path/to/prefix
```

Configure the consumer with `-DCMAKE_PREFIX_PATH=/path/to/prefix` if the
prefix is outside CMake's default search paths. In the consumer, find the
package and link its imported target.

```cmake
find_package(vqm CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE vqm::core)
```

With `add_subdirectory`, link the same target. The CLI, benchmark and tests
default to off in an embedded build. The core links the Vulkan loader without
requiring libpng or libjpeg. The installed core is a static C++ library that
can also link into a shared library. Use compatible C++ compiler and standard
library settings for the core and consumer.

Use `vqm::ListDevices()` from `<vqm_device.h>` to discover compatible
compute devices. Pass the chosen UUID to `vqm::SetDefaultDevice(uuid)`
before the first operation that uses the library-owned device. This choice
applies to default pipelines, pools, and memory queries in the process.
Applications managing their own Vulkan devices can select a device per
pipeline with the existing `VulkanDevice` overloads.

```cpp
#include <vqm_metrics.h>

// Populate these with decoded, planar pixels and their format descriptors.
vqm::Image original, distorted;
vqm::MetricsConfig config;
config.size = original.size;
config.original_format = original.format;
config.distorted_format = distorted.format;
config.ssimulacra2 = vqm::Ssimulacra2Config{};
config.butteraugli = vqm::ButteraugliConfig{};
vqm::Metrics metrics(config);
auto scores = metrics.ComputeImage(original, distorted);
double ssimu2 = *scores.ssimulacra2;
double butter = *scores.butteraugli;
```

The [unified metrics guide](metrics.md) covers VMAF, video sequences,
external buffers, and memory planning. Lower-level pipelines remain available.
A pipeline has a fixed geometry and is used by one host thread. It also
supports uploaded images, borrowed host planes and caller-owned Vulkan buffers.
`ButteraugliPipeline` offers the corresponding API for Butteraugli. For batch
work, `Ssimulacra2Pool` keeps a bounded number of pipelines in flight. The core
API evaluates one configured alpha background; `ComputeWorstBackground` in
`vqm_alpha.h` implements the CLI's policy.
`VmafPipeline` in `vmaf.h` accepts ordered frame pairs and emits scores with
the motion look-ahead described in the [VMAF guide](vmaf.md).
`VmafConfig::model` selects the 1080 or 4K predictor; `neg` and `transform`
toggle the feature-gain and phone-score options, and `fit`/`model_size` select
the evaluation geometry. `temporal_weighting = false` is a derived
mode for content where motion2 is misleading: it forces motion2 to zero, skips
the motion passes, and scores every frame immediately.
`VmafV1Pipeline` in `vmaf_v1.h` selects a viewing preset through
`VmafV1Config::model`. Its `fit` setting can scale or pad luma and chroma.
Set `VmafV1Config::chroma_sampling` to
`VmafChromaSampling::Yuv420` or `Yuv444` to convert SDR inputs to a common
BT.709 representation, including mixed source formats; the default `Native`
mode preserves compatible coded YUV. The benchmark exposes the same selection
as `--vmaf-chroma native|420|444`.

See the [API and implementation guide](reference.md#core-api) for input
layout, lifetime and synchronization requirements, and
[Vulkan synchronization](vulkan-synchronization.md) for interoperation
with other Vulkan work. [GPU memory planning](memory.md) describes how to
query per-metric buffer requirements and estimate worker counts before allocating.
[Shared workspaces](shared-workspace.md) let SSIMULACRA 2 and Butteraugli
reuse an arena when they run sequentially.
