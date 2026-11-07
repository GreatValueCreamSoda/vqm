// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.

#include "metrics/butteraugli/layout.h"

#include <limits>
#include <stdexcept>

namespace vqm::internal::butteraugli_layout {
namespace {

std::uint64_t DispatchGroups(std::uint64_t extent, std::uint32_t tile) {
    return extent / tile + (extent % tile != 0);
}

void CheckDispatch(
    const VkPhysicalDeviceLimits& limits, std::uint64_t x, std::uint64_t y,
    std::uint64_t z
) {
    if (x > limits.maxComputeWorkGroupCount[0] ||
        y > limits.maxComputeWorkGroupCount[1] ||
        z > limits.maxComputeWorkGroupCount[2]) {
        throw std::invalid_argument(
            "butter: dispatch exceeds device workgroup count limits"
        );
    }
}

// The shader receives one row per level. Offsets address the shared arena;
// subsampled RGB is produced at level 0 and consumed at level 1.
void PopulateLevelTable(Layout& layout, const ButteraugliGeometry& geometry) {
    const auto sub_width = static_cast<std::uint32_t>(geometry.sub.width);
    const auto sub_height = static_cast<std::uint32_t>(geometry.sub.height);
    for (std::uint32_t level = 0; level < layout.levels; ++level) {
        std::uint32_t* row = layout.table.data() + level * kRowWords;
        const LevelLayout& level_layout = layout.level[level];
        row[R_CHAIN0] = level_layout.chain[0];
        row[R_CHAIN1] = level_layout.chain[1];
        row[R_SCRATCH_A] = level_layout.scratch_a;
        row[R_SCRATCH_B] = level_layout.scratch_b;
        row[R_AC] = level_layout.ac;
        row[R_DIFFMAP] = level_layout.diffmap;
        row[R_SUB0] = layout.sub0;
        row[R_SUB1] = layout.sub1;
        row[R_WIDTH] = static_cast<std::uint32_t>(level_layout.size.width);
        row[R_HEIGHT] = static_cast<std::uint32_t>(level_layout.size.height);
        row[R_AREA] = level_layout.area;
        row[R_SUBW] = sub_width;
        row[R_SUBH] = sub_height;
        if (level == 0) row[R_PARTIALS] = layout.partials;
    }
}

// Reserve source uploads first, then the descriptors, lookup tables, and
// reduction readback needed when the arena is not host-visible.
void PlanButterStaging(
    ButterLayoutPlan& plan, bool streaming, std::size_t resident_images
) {
    if (plan.base_pixels >
        (std::numeric_limits<std::size_t>::max() - 1024u) / 8u)
        throw std::invalid_argument("butter: staging exceeds host addressing");
    plan.staging =
        PlanStagingSources(plan.base_pixels, streaming, resident_images);
    plan.staging_desc = plan.staging.reserved_floats;
    plan.staging_kernels = plan.staging_desc + 2 * kSourceDescWords;
    plan.staging_table = plan.staging_kernels + kKernelFloats;
    plan.staging_partials = plan.staging_table + kTableWords;
    const std::uint64_t staging_words =
        plan.staging_partials +
        4u * static_cast<std::uint64_t>(plan.layout.partial_groups);
    if (staging_words > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("butter: staging exceeds host addressing");
}

}  // namespace

ButterLayoutPlan BuildButterLayout(
    const ButteraugliGeometry& geometry, const VulkanContext& context,
    bool streaming, std::size_t resident_images, std::size_t input_bytes,
    ChromaFilter chroma_filter
) {
    if (chroma_filter != ChromaFilter::Bilinear &&
        chroma_filter != ChromaFilter::Bicubic &&
        chroma_filter != ChromaFilter::Lanczos)
        throw std::invalid_argument("butter: invalid chroma filter");
    // butter_opsin.comp uses three 36x36 source tiles and three 36x32
    // intermediate tiles. The L2 and reduction shaders use 256 lanes along X.
    constexpr std::size_t kOpsinSharedBytes =
        3 * (36 * 36 + 36 * 32) * sizeof(float);
    if (context.limits.maxComputeWorkGroupSize[0] < 256 ||
        context.limits.maxComputeSharedMemorySize < kOpsinSharedBytes)
        throw std::invalid_argument(
            "butter: requires a 256-wide workgroup and 29376 bytes of "
            "shared memory"
        );
    if (geometry.base.width == 0 || geometry.base.height == 0 ||
        geometry.base.width > std::numeric_limits<std::uint32_t>::max() ||
        geometry.base.height > std::numeric_limits<std::uint32_t>::max() ||
        geometry.base.width >
            std::numeric_limits<std::uint32_t>::max() / geometry.base.height)
        throw std::invalid_argument(
            "butter: image geometry exceeds arena addressing"
        );
    ButterLayoutPlan plan;
    auto& layout = plan.layout;
    const std::size_t base_pixels =
        static_cast<std::size_t>(geometry.base.width) * geometry.base.height;
    plan.base_pixels = base_pixels;
    layout.levels = geometry.levels;
    layout.sub_area =
        static_cast<std::uint32_t>(geometry.sub.width * geometry.sub.height);

    std::uint64_t cursor = 0;
    const auto add = [&cursor](std::uint64_t floats) {
        constexpr auto max = std::numeric_limits<std::uint32_t>::max();
        if (floats > max - 15u)
            throw std::invalid_argument(
                "butter: geometry exceeds arena addressing"
            );
        const std::uint64_t aligned = (floats + 15u) / 16u * 16u;
        if (aligned > max - cursor)
            throw std::invalid_argument(
                "butter: geometry exceeds arena addressing"
            );
        const auto offset = static_cast<std::uint32_t>(cursor);
        cursor += aligned;
        return offset;
    };
    const std::uint32_t kernels = add(kKernelFloats);
    const std::uint32_t table = add(kTableWords);
    if (kernels != 0) throw std::logic_error("butter: kernel table misplaced");
    layout.table_word = table;

    for (std::uint32_t level = 0; level < layout.levels; ++level) {
        LevelLayout& out = layout.level[level];
        out.size = level == 0 ? geometry.base : geometry.sub;
        out.area = static_cast<std::uint32_t>(out.size.width * out.size.height);
        // Opsin uses ceil(ceil(extent/2)/16), the tiled passes use
        // ceil(extent/32), and the linear/reduce passes use ceil(area/256).
        CheckDispatch(
            context.limits, DispatchGroups(out.size.width, 32),
            DispatchGroups(out.size.height, 32), 3
        );
        CheckDispatch(context.limits, DispatchGroups(out.area, 256), 1, 1);
        const std::uint64_t planes = out.area;
        out.chain[0] = add(kChainPlanes * planes);
        out.chain[1] = add(kChainPlanes * planes);
        // The distorted chain is rebuilt on every call. Opsin and the LF
        // blur use planes 6..8 as scratch A; the LF and HF blurs use 9..11
        // as scratch B. HF writes only planes 6 and 7 as auxiliary output
        // (its third channel has no HF). UHF then reads those two planes and
        // writes planes 8..11 after the HF dispatch has finished.
        //
        // Once Malta and L2 have consumed the final HF and UHF planes, L2
        // writes the two masks into scratch A planes 6 and 7 and the mask
        // blur writes scratch B planes 9 and 10. Each dispatch is followed by
        // a compute barrier. The reference chain stays intact for caching.
        out.scratch_a = out.chain[1] + 6u * out.area;
        out.scratch_b = out.chain[1] + 9u * out.area;
        // Diff levels run sequentially, so their three AC planes can share
        // the full-size allocation. The coarse diffmap is written after the
        // mask blur has finished reading scratch A, and remains there until
        // the full-size erosion folds it into its own separate diffmap.
        out.ac = level == 0 ? add(3 * planes) : layout.level[0].ac;
        out.diffmap = level == 0 ? add(planes) : out.scratch_a;
    }
    // Each level-0 subsample is consumed by level-1 opsin before the next
    // chain is built. AC is first used after both chains are complete, so the
    // two temporary RGB images can share its larger level-0 allocation.
    layout.sub0 = layout.level[0].ac;
    layout.sub1 = layout.level[0].ac;
    layout.partial_groups =
        static_cast<std::uint32_t>(DispatchGroups(layout.level[0].area, 256));
    layout.partials =
        add(4u * static_cast<std::uint64_t>(layout.partial_groups));
    layout.final_result = add(4);
    const std::uint64_t input_floats = input_bytes / 4 + (input_bytes % 4 != 0);
    layout.input_pool = add(input_floats);
    layout.input_pool_size = static_cast<std::uint32_t>(input_floats);
    for (std::uint32_t image = 0; image < 2; ++image) {
        // The descriptors are needed for every input kind, but only streaming
        // submissions need an arena region for copied-in source pixels.
        layout.stream_region[image] =
            streaming ? add(kSourceFloatsPerPixel *
                            static_cast<std::uint64_t>(base_pixels))
                      : 0;
        layout.stream_desc[image] = add(kSourceDescWords);
    }
    layout.resident.reserve(resident_images);
    for (std::size_t slot = 0; slot < resident_images; ++slot) {
        layout.resident.push_back(
            add(kSourceFloatsPerPixel * static_cast<std::uint64_t>(base_pixels))
        );
    }
    if (cursor > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::invalid_argument("butter: arena exceeds host addressing");
    layout.bytes = Bytes(static_cast<std::size_t>(cursor));
    if (layout.bytes >
        static_cast<VkDeviceSize>(context.limits.maxStorageBufferRange)) {
        throw std::invalid_argument(
            "image geometry needs more storage buffer than this GPU supports"
        );
    }

    PopulateLevelTable(layout, geometry);
    PlanButterStaging(plan, streaming, resident_images);
    return plan;
}

}  // namespace vqm::internal::butteraugli_layout

namespace vqm::internal {

BufferMemoryPlan PlanButteraugliMemory(
    const ButteraugliConfig& config, const VulkanContext& context
) {
    ValidateButteraugliConfig(config);
    const auto geometry = PlanButteraugliGeometry(config.size);
    const auto plan = butteraugli_layout::BuildButterLayout(
        geometry, context, config.streaming_inputs, config.gpu_resident_images,
        config.input_bytes, config.chroma_filter
    );
    return {static_cast<std::size_t>(plan.layout.bytes),
            static_cast<std::size_t>(
                Bytes(plan.staging_partials + 4u * plan.layout.partial_groups)
            )};
}

}  // namespace vqm::internal
