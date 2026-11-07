// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "metrics/ssimulacra2/layout.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "handles.h"

namespace vqm::internal::ssimulacra2_layout {
namespace {

// Output tiles src/metrics/ssimulacra2/shaders/scale.comp is built for: the tile
// in output pixels, the local size factor (the workgroup is kWorkgroup x lanes,
// so kWorkgroup * lanes invocations), and the shared-memory footprint in bytes
// the shader describes, 2*(cols+12)*(rows+12) + (rows+12)*cols floats. An
// invocation owns (rows*cols)/(kWorkgroup*lanes) output rows -- four; the
// shader's #if enforces that, so a bad entry fails the compile. The
// pipeline picks the first variant the device allows, so the geometry follows
// VkPhysicalDeviceLimits instead of one GPU's numbers.
//
// Membership and order are measured, not guessed: a wider or taller tile reads
// less halo per output but runs a coarser workgroup, and more lanes buy warps
// for the same tile at the price of idle invocations in the horizontal pass,
// whose work is cols/4 * (rows+12) four-column windows. Both effects were
// measured on the scale stage at 4K (resident, median of 20 iterations, three
// runs each, 32 x 32 unless noted): 16 x 32 with 128 invocations 1.72 ms, the
// 32 x 32 default 1.54 ms, a 32 x 64 tile (38 KB, 512 invocations) 1.65 ms, and
// 512 invocations for the same 32 x 32 tile -- two output rows per invocation,
// so 14 shared loads feed 26 FMAs instead of 16 feeding 52 -- 2.48 ms. Depth
// per invocation is not what limits the stage: halving the rows takes the
// shared loads per FMA from 0.31 to 0.54, and the stage is bound by
// shared-memory instructions.
//
// Every variant multiplies its column count by four in the horizontal pass, so
// cols must be a multiple of four, and scale.comp's #if requires exactly four
// output rows per invocation.
// Order and membership match VQM_SCALE_VARIANTS in CMakeLists.txt,
// which compiles one scale.comp blob per entry; both lists change together.
constexpr ScaleVariant kScaleVariants[] = {
    {32u, 32u, 16u, 21120u},  // 256, four rows
    {16u, 32u, 8u, 13440u},   // 128, four rows
};
constexpr std::uint32_t Threads(const ScaleVariant& variant) {
    return kWorkgroup * variant.lanes;
}

// The first tile in kScaleVariants the device can run: shared memory,
// invocations per workgroup and the workgroup shape all come from the driver.
// The list is ordered best-first, so the first fit wins; its last entry (16 x
// 32, the Vulkan minimum of 16 KiB and 128 invocations) fits every device
// Inspect() lets through, so the loop never falls off the end. A device that
// fits no entry would break that invariant, so the impossible path aborts
// rather than pretending some variant works.
const ScaleVariant& PickScaleVariant(const VkPhysicalDeviceLimits& limits) {
    for (const ScaleVariant& variant : kScaleVariants) {
        if (variant.shared_bytes > limits.maxComputeSharedMemorySize) continue;
        if (Threads(variant) > limits.maxComputeWorkGroupInvocations) continue;
        if (limits.maxComputeWorkGroupSize[0] < kWorkgroup) continue;
        if (limits.maxComputeWorkGroupSize[1] < variant.lanes) continue;
        return variant;
    }
    std::abort();
}

std::uint32_t AddAlignedWords(std::uint64_t& cursor, std::uint64_t count) {
    constexpr auto max = std::numeric_limits<std::uint32_t>::max();
    if (count > max - 15u)
        throw std::invalid_argument(
            "vulkan: image geometry exceeds arena addressing"
        );
    const std::uint64_t aligned = (count + 15u) / 16u * 16u;
    if (aligned > max - cursor)
        throw std::invalid_argument(
            "vulkan: image geometry exceeds arena addressing"
        );
    const auto offset = static_cast<std::uint32_t>(cursor);
    cursor += aligned;
    return offset;
}

std::uint64_t DispatchGroups(std::size_t extent, std::uint32_t tile) {
    const auto value = static_cast<std::uint64_t>(extent);
    return value / tile + (value % tile != 0);
}

void CheckDispatch(
    const VkPhysicalDeviceLimits& limits, std::uint64_t x, std::uint64_t y,
    std::uint64_t z
) {
    if (x > limits.maxComputeWorkGroupCount[0] ||
        y > limits.maxComputeWorkGroupCount[1] ||
        z > limits.maxComputeWorkGroupCount[2]) {
        throw std::invalid_argument(
            "ssimulacra2: dispatch exceeds device workgroup count limits"
        );
    }
}

}  // namespace

Ssimulacra2LayoutPlan BuildSsimulacra2Layout(
    const std::vector<Size>& sizes, const VulkanContext& context,
    std::size_t resident_images, bool half_planes, bool streaming,
    std::size_t input_bytes, ChromaFilter chroma_filter
) {
    if (sizes.empty() || sizes.size() > kSsimulacra2MaxScales)
        throw std::invalid_argument("vulkan backend needs valid scales");
    if (resident_images > kResidentIndexMask)
        throw std::invalid_argument("gpu_resident_images exceeds slot limit");
    if (chroma_filter != ChromaFilter::Bilinear &&
        chroma_filter != ChromaFilter::Bicubic &&
        chroma_filter != ChromaFilter::Lanczos)
        throw std::invalid_argument("ssimulacra2: invalid chroma filter");
    // reduce.comp needs 256 invocations along X even with the smaller scale
    // tile. The context checks total invocations independently of this limit.
    if (context.limits.maxComputeWorkGroupSize[0] < 256)
        throw std::invalid_argument(
            "ssimulacra2: reduction requires a 256-wide workgroup"
        );
    if (half_planes && !context.sixteen_bit_storage)
        throw std::runtime_error(
            "vulkan: device does not support 16-bit storage; "
            "Ssimulacra2Config::half_planes needs it"
        );
    for (const auto& size : sizes) {
        if (size.width == 0 || size.height == 0 ||
            size.width > std::numeric_limits<std::uint32_t>::max() ||
            size.height > std::numeric_limits<std::uint32_t>::max() ||
            size.width > std::numeric_limits<std::uint32_t>::max() / size.height)
            throw std::invalid_argument(
                "vulkan: image geometry exceeds arena addressing"
            );
    }
    Ssimulacra2LayoutPlan out;
    auto& regions = out.regions;
    auto& scale_tile = out.scale_tile;
    auto& base_pixels = out.base_pixels;
    auto& levels = out.levels;
    auto& level_groups = out.level_groups;
    auto& level_partials = out.level_partials;
    auto& staging_norm_offset = out.staging_norm_offset;
    auto& staging_desc_offset = out.staging_desc_offset;
    auto& staging_level_offset = out.staging_level_offset;
    base_pixels = sizes[0].width * sizes[0].height;
    if (base_pixels > (std::numeric_limits<std::size_t>::max() - 1024u) / 8u)
        throw std::invalid_argument("vulkan: staging exceeds host addressing");
    scale_tile = PickScaleVariant(context.limits);
    levels = sizes.size();
    const std::size_t base = base_pixels;

    // Check every shape the prepare, scale and reduce dispatches use. The
    // division-first form cannot overflow for a large but valid size.
    const auto half_up = [](std::size_t value) { return value / 2 + value % 2; };
    CheckDispatch(
        context.limits, DispatchGroups(half_up(sizes[0].width), kWorkgroup),
        DispatchGroups(half_up(sizes[0].height), kWorkgroup), 1
    );
    CheckDispatch(context.limits, kScaleSums, levels, 1);

    for (std::size_t level = 0; level < levels; ++level) {
        regions.channel_floats[level] =
            static_cast<std::uint32_t>(sizes[level].width * sizes[level].height);
    }

    // The plane sets, in plane elements. Binding 1 sees only this region,
    // so its offsets start at 0 like the old arena ones did. One set per
    // image per level holds that level's XYB planes; the linear samples the
    // pyramid averages between levels never reach this region, so it holds
    // exactly what the scale stage reads.
    std::uint64_t plane_cursor = 0;
    const auto add_plane = [&plane_cursor](std::uint64_t elements) {
        return AddAlignedWords(plane_cursor, elements);
    };
    for (std::size_t image = 0; image < 2; ++image) {
        for (std::size_t level = 0; level < levels; ++level) {
            regions.img[image][level] = add_plane(
                kChannels *
                static_cast<std::uint64_t>(regions.channel_floats[level])
            );
        }
    }

    // The f32 regions, in arena floats. Each starts on a 16-float (64 B)
    // boundary.
    std::uint64_t cursor = 0;
    const auto add = [&cursor](std::uint64_t floats) {
        return AddAlignedWords(cursor, floats);
    };
    // The streamed input stays in its raw sample layout: at most four
    // planes of at most four bytes per sample, so 16 bytes per pixel
    // covers every accepted format.
    for (std::size_t image = 0; image < 2; ++image) {
        regions.source[image] =
            streaming
                ? add(kSourceFloatsPerPixel * static_cast<std::uint64_t>(base))
                : 0;
    }
    // The per-image constant block the prepare shader reads.
    for (std::size_t image = 0; image < 2; ++image) {
        regions.source_desc[image] = add(kSourceDescWords);
    }
    // The pool caller-owned image buffers are carved from.
    const std::size_t input_floats = input_bytes / 4 + (input_bytes % 4 != 0);
    out.input_pool = add(input_floats);
    out.input_pool_size = static_cast<std::uint32_t>(input_floats);
    // The plane geometry every dispatch reads: the per-image plane
    // offsets, the per-level norm blocks, the level sizes and the scale
    // grid.
    regions.levels = add(kLevelWords);
    // One arena region per resident image slot, same capacity.
    regions.resident.reserve(resident_images);
    for (std::size_t slot = 0; slot < resident_images; ++slot) {
        regions.resident.push_back(
            add(kSourceFloatsPerPixel * static_cast<std::uint64_t>(base))
        );
    }
    // One block of norm partials per level, packed in level order: a
    // level's reduction reads only its own block, and every level's block
    // is live at once because one reduction covers them all.
    std::uint64_t partial_floats = 0;
    for (std::size_t level = 0; level < levels; ++level) {
        const auto gx = DispatchGroups(sizes[level].width, scale_tile.cols);
        const auto gy = DispatchGroups(sizes[level].height, scale_tile.rows);
        CheckDispatch(context.limits, gx, gy, 3);
        const auto groups = gx * gy;
        if (groups > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument(
                "ssimulacra2: scale groups exceed arena addressing"
            );
        level_groups[level] = static_cast<std::uint32_t>(groups);
        if (partial_floats > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument(
                "vulkan: partials exceed arena addressing"
            );
        level_partials[level] = static_cast<std::uint32_t>(partial_floats);
        partial_floats +=
            static_cast<std::uint64_t>(kScaleSums) * level_groups[level];
    }
    regions.partials = add(partial_floats);
    regions.norms = add(levels * kScaleSums);

    // The plane region closes the buffer, on an offset the descriptor
    // needs aligned: binding 1 points at it, and its byte offset must be
    // a multiple of minStorageBufferOffsetAlignment. The cursor counts
    // floats, though, so the boundary is rounded to whole floats first:
    // `step` is plane_align / gcd(plane_align, 4), the coarsest float
    // count whose byte size still lands on the boundary (4 floats for a
    // 16-byte alignment, 1 float for a 4-byte one). Without it a
    // float-indexed cursor could stop short of the aligned byte.
    const std::uint64_t plane_align = std::max<std::uint64_t>(
        64u, context.limits.minStorageBufferOffsetAlignment
    );
    const std::uint64_t step =
        plane_align / std::gcd(plane_align, sizeof(float));
    if (step > std::numeric_limits<std::uint32_t>::max() ||
        cursor > std::numeric_limits<std::uint32_t>::max() - (step - 1))
        throw std::invalid_argument(
            "vulkan: plane alignment exceeds arena addressing"
        );
    cursor = (cursor + step - 1) / step * step;
    if (cursor > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("vulkan: arena exceeds host addressing");
    regions.plane_base_bytes = Bytes(static_cast<std::size_t>(cursor));
    regions.plane_floats = static_cast<std::uint32_t>(plane_cursor);
    regions.plane_bytes = PlaneBytes(regions.plane_floats, half_planes);

    regions.bytes = regions.plane_base_bytes + regions.plane_bytes;
    if (regions.bytes >
        static_cast<VkDeviceSize>(context.limits.maxStorageBufferRange)) {
        throw std::invalid_argument(
            "image geometry needs more storage buffer than this GPU "
            "supports"
        );
    }

    // The staging offsets are a property of this layout: the readback
    // lands after the two images' upload slots, then the descriptors, then
    // the level table. A non-streaming pipeline still reserves one image at
    // word 0 when it has resident slots, because UploadResident stages a
    // resident upload there on a device whose arena is not host visible;
    // the readback then starts one image in, which is why it cannot land on
    // the upload.
    out.staging = PlanStagingSources(base_pixels, streaming, resident_images);
    const auto& staging = out.staging;
    staging_norm_offset = staging.reserved_floats;
    staging_desc_offset = staging_norm_offset + sizes.size() * kScaleSums;
    staging_level_offset = staging_desc_offset + 2 * kSourceDescWords;
    if (staging_level_offset + kLevelWords >
        std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("vulkan: staging exceeds host addressing");

    return out;
}

}  // namespace vqm::internal::ssimulacra2_layout

namespace vqm::internal {

BufferMemoryPlan PlanSsimulacra2Memory(
    const Ssimulacra2Config& config, const VulkanContext& context
) {
    const auto scales = PlanSsimulacra2Scales(config.size);
    const std::vector<Size> sizes(
        scales.sizes.begin(), scales.sizes.begin() + scales.count
    );
    const auto layout = ssimulacra2_layout::BuildSsimulacra2Layout(
        sizes, context, config.gpu_resident_images, config.half_planes,
        config.streaming_inputs, config.input_bytes, config.chroma_filter
    );
    return {static_cast<std::size_t>(layout.regions.bytes),
            static_cast<std::size_t>(Bytes(
                layout.staging_level_offset + ssimulacra2_layout::kLevelWords
            ))};
}

}  // namespace vqm::internal
