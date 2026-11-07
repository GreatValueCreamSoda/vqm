# CVVDP memory qualification

`memory_probe.cc` compares queried Vulkan buffer requirements to actual
allocations at 1080p and 4K, each at 24/30/60/120 fps. It counts individual
temporal allocations, including their memory type and heap, and constructs only
one pipeline at a time. Device-limit rejection is reported separately from
allocation failures, which fail the process. A successful query does not promise
enough available VRAM. The 4K120 case can require roughly 7 GiB.

From the repository root, after a release build:

```sh
c++ -std=c++20 -O2 -Iinclude bench/experiments/cvvdp/memory_probe.cc \
  build/libvqm_core.a -lvulkan -o build/cvvdp_memory_probe
./build/cvvdp_memory_probe
```

Use a separate machine/device-specific result file under `artifacts/reports/`.
The reference method, original numerical oracles and compatibility limits are
documented in [CVVDP development](../../../docs/cvvdp-development.md).
