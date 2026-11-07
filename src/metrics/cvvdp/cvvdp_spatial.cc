// ColorVideoVDP: Copyright (c) 2024, Graphics and Displays group.
// MIT; see licenses/COLORVIDEOVDP-LICENSE.
// Spatial stages follow the original v0.5.7 (2a268bce), pycvvdp/lpyr_dec.py
// and cvvdp_metric.py. GPU pass boundaries and buffer reuse are VQM's.

#include "cvvdp_spatial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "cvvdp_model_data.inc"
#include "vulkan/vulkan_runtime.h"

namespace vqm::internal {
namespace {

constexpr std::uint32_t kLanes = 256;
constexpr std::size_t kMaxChannels = 4;
constexpr std::size_t kTableHeader = 40;
constexpr std::size_t kCsfSamples = 32;

enum PipelineIndex : std::size_t { kPyramid, kContrast, kMask, kReduce };

struct SpatialPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t next_width;
    std::uint32_t next_height;
    std::uint32_t reference;
    std::uint32_t distorted;
    std::uint32_t destination;
    std::uint32_t auxiliary;
    std::uint32_t scratch;
    std::uint32_t tables;
    std::uint32_t partials;
    std::uint32_t results;
    std::uint32_t channels;
    std::uint32_t bands;
    std::uint32_t band;
    std::uint32_t op;
    std::uint32_t input_stride;
    std::uint32_t destination_stride;
    std::uint32_t groups;
    std::uint32_t parity;
};
static_assert(sizeof(SpatialPush) == CvvdpSpatial::PushBytes());

std::uint32_t Word(std::size_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("cvvdp: spatial arena offset overflow");
    return static_cast<std::uint32_t>(count);
}

std::size_t Pixels(Size size) {
    constexpr std::size_t maximum =
        std::numeric_limits<std::uint32_t>::max() / 32;
    if (size.width < 2 || size.height < 2 || size.width > maximum ||
        size.height > maximum / size.width)
        throw std::invalid_argument("cvvdp: invalid spatial geometry");
    return size.width * size.height;
}

struct Level {
    Size size;
    std::uint32_t reference = 0;
    std::uint32_t distorted = 0;
};

// The two original input planes are borrowed as level zero. Coarser Gaussian
// levels remain live until all bands have been scored. One temporary region
// serves reduction/expansion intermediates, then difference responses; another
// serves expanded backgrounds, then Gaussian masking intermediates. Mutual
// contrast occupies a third region. No per-band response maps are retained.
struct SpatialLayout {
    std::vector<Level> levels;
    std::uint32_t temporary = 0;
    std::uint32_t expanded = 0;
    std::uint32_t mutual = 0;
    std::uint32_t words = 0;

    SpatialLayout(Size size, double ppd, std::uint32_t start) {
        const std::size_t pixels = Pixels(size);
        const std::size_t count = CvvdpSpatial::Levels(size, ppd);
        std::size_t next = start;
        std::size_t temporary_words = kMaxChannels * pixels;
        levels.push_back({size});
        for (std::size_t band = 1; band < count; ++band) {
            const Size coarse{(size.width + 1) / 2, (size.height + 1) / 2};
            temporary_words = std::max(
                {temporary_words, 2 * kMaxChannels * size.width * coarse.height,
                 2 * kMaxChannels * coarse.width * size.height}
            );
            const auto reference = Word(next);
            next += kMaxChannels * Pixels(coarse);
            const auto distorted = Word(next);
            next += kMaxChannels * Pixels(coarse);
            levels.push_back({coarse, reference, distorted});
            size = coarse;
        }
        temporary = Word(next);
        next += temporary_words;
        expanded = Word(next);
        next += 2 * kMaxChannels * pixels;
        mutual = Word(next);
        next += kMaxChannels * pixels;
        words = Word(next - start);
        (void)Word(next);
    }
};

float BandFrequency(std::size_t band, std::size_t count, double ppd) {
    if (band + 1 == count) return 0.1f;
    if (band == 0) return static_cast<float>(ppd / 2.0);
    return static_cast<float>(
        0.3228 * std::exp2(-static_cast<double>(band - 1)) *
        static_cast<double>(ppd) / 2.0
    );
}

void FillCsfRow(float rho, std::size_t channel, std::span<float> row) {
    using namespace cvvdp_data;
    const float query = std::log10(rho);
    const auto upper = std::lower_bound(log_rho.begin(), log_rho.end(), query);
    const auto index = static_cast<std::size_t>(upper - log_rho.begin());
    const std::size_t lower =
        std::clamp(index, std::size_t{1}, kCsfSamples - 1) - 1;
    for (std::size_t luminance = 0; luminance < kCsfSamples; ++luminance) {
        const std::size_t offset =
            (channel * kCsfSamples + luminance) * kCsfSamples + lower;
        // Match batch_interp1d's subtraction/division/multiplication order.
        const float slope =
            (log_sensitivity[offset + 1] - log_sensitivity[offset]) /
            (log_rho[lower + 1] - log_rho[lower]);
        row[luminance] =
            log_sensitivity[offset] + slope * (query - log_rho[lower]);
    }
}

void RecordPyramidPass(
    VulkanRuntime& runtime, VkPipeline pipeline, SpatialPush& push,
    const Level& input, const Level& output, std::uint32_t op,
    std::uint32_t parity
) {
    push.width = Word(input.size.width);
    push.height = Word(input.size.height);
    push.next_width = Word(output.size.width);
    push.next_height = Word(output.size.height);
    push.reference = input.reference;
    push.distorted = input.distorted;
    push.destination = output.reference;
    push.auxiliary = output.distorted;
    push.input_stride = Word(Pixels(input.size));
    push.destination_stride = Word(Pixels(output.size));
    push.op = op;
    push.parity = parity;
    runtime.Dispatch(
        pipeline, &push, sizeof(push), Groups(Pixels(output.size), kLanes),
        2 * push.channels, 1
    );
    runtime.ComputeBarrier();
}

void RecordGaussianPyramid(
    VulkanRuntime& runtime, VkPipeline pipeline, SpatialPush& push,
    const SpatialLayout& layout
) {
    for (std::size_t band = 1; band < layout.levels.size(); ++band) {
        const auto& input = layout.levels[band - 1];
        const auto& output = layout.levels[band];
        const Size vertical_size{input.size.width, output.size.height};
        const Level vertical{
            vertical_size, layout.temporary,
            layout.temporary + push.channels * Word(Pixels(vertical_size))};
        RecordPyramidPass(
            runtime, pipeline, push, input, vertical, 0,
            Word(input.size.height % 2)
        );
        RecordPyramidPass(
            runtime, pipeline, push, vertical, output, 1,
            Word(input.size.height % 2)
        );
    }
}

void RecordExpandedBand(
    VulkanRuntime& runtime, VkPipeline pipeline, SpatialPush& push,
    const SpatialLayout& layout, std::size_t band
) {
    const auto& level = layout.levels[band];
    const auto& coarse = layout.levels[band + 1];
    const Size vertical_size{coarse.size.width, level.size.height};
    const Level vertical{
        vertical_size, layout.temporary,
        layout.temporary + push.channels * Word(Pixels(vertical_size))};
    const Level expanded{
        level.size, layout.expanded,
        layout.expanded + push.channels * Word(Pixels(level.size))};
    RecordPyramidPass(runtime, pipeline, push, coarse, vertical, 2, 0);
    RecordPyramidPass(runtime, pipeline, push, vertical, expanded, 3, 0);
}

void RecordMask(
    VulkanRuntime& runtime, VkPipeline pipeline, SpatialPush& push,
    const SpatialLayout& layout, bool baseband
) {
    push.groups = Groups(Pixels({push.width, push.height}), kLanes);
    if (!baseband) {
        push.reference = layout.mutual;
        push.destination = layout.expanded;
        if (push.width > 6 && push.height > 6) {
            push.op = 0;
            runtime.Dispatch(
                pipeline, &push, sizeof(push), push.groups, push.channels, 1
            );
            runtime.ComputeBarrier();
            push.reference = layout.expanded;
            push.destination = layout.mutual;
            push.op = 1;
        } else {
            push.op = 2;
        }
        runtime.Dispatch(
            pipeline, &push, sizeof(push), push.groups, push.channels, 1
        );
        runtime.ComputeBarrier();
        push.distorted = push.destination;
    }
    push.reference = layout.temporary;
    push.op = baseband ? 4u : 3u;
    runtime.Dispatch(
        pipeline, &push, sizeof(push), push.groups, push.channels, 1
    );
    runtime.ComputeBarrier();
}

}  // namespace

std::size_t CvvdpSpatial::Levels(Size size, double pixels_per_degree) {
    (void)Pixels(size);
    if (!std::isfinite(pixels_per_degree) || pixels_per_degree <= 0.0f)
        throw std::invalid_argument("cvvdp: pixels per degree must be positive");
    const auto maximum = static_cast<std::size_t>(
        std::floor(
            std::log2(static_cast<double>(std::min(size.width, size.height)))
        ) -
        1.0
    );
    std::size_t band = maximum;
    for (std::size_t index = 0; index < 15; ++index) {
        const double frequency =
            index == 0 ? pixels_per_degree / 2.0
                       : 0.3228 * std::exp2(-static_cast<double>(index - 1)) *
                             pixels_per_degree / 2.0;
        if (frequency <= 0.2) {
            band = index;
            break;
        }
    }
    return std::min(band + 1, maximum) + 1;
}

std::size_t CvvdpSpatial::ScratchWords(Size size, double pixels_per_degree) {
    return SpatialLayout(size, pixels_per_degree, 0).words;
}

std::size_t CvvdpSpatial::TableWords(Size size, double pixels_per_degree) {
    return kTableHeader +
           Levels(size, pixels_per_degree) * kMaxChannels * kCsfSamples;
}

std::size_t CvvdpSpatial::PartialCount(Size size) {
    return (Pixels(size) + kLanes - 1) / kLanes;
}

void CvvdpSpatial::FillTables(
    Size size, double pixels_per_degree, std::span<float> destination
) {
    using namespace cvvdp_data;
    if (destination.size() != TableWords(size, pixels_per_degree))
        throw std::invalid_argument("cvvdp: incorrect spatial table size");
    std::fill(destination.begin(), destination.end(), 0.0f);
    destination[0] = mask_p;
    destination[1] = std::pow(10.0f, mask_c);
    destination[2] = std::pow(10.0f, d_max);
    destination[3] = std::pow(10.0f, sensitivity_correction / 20.0f);
    std::copy(mask_q.begin(), mask_q.end(), destination.begin() + 4);
    std::transform(
        xcm_weights.begin(), xcm_weights.end(), destination.begin() + 8,
        [](float value) { return std::exp2(value); }
    );
    float total = 0.0f;
    for (int tap = -6; tap <= 6; ++tap) {
        const float coordinate = static_cast<float>(tap);
        const float weight = std::exp(-0.5f * coordinate * coordinate / 9.0f);
        destination[static_cast<std::size_t>(24 + tap + 6)] = weight;
        total += weight;
    }
    for (std::size_t tap = 24; tap < 37; ++tap) destination[tap] /= total;
    destination[37] = log_luminance.front();
    destination[38] = log_luminance.back();
    const std::size_t bands = Levels(size, pixels_per_degree);
    for (std::size_t band = 0; band < bands; ++band) {
        for (std::size_t channel = 0; channel < kMaxChannels; ++channel) {
            const std::size_t row =
                kTableHeader + (band * kMaxChannels + channel) * kCsfSamples;
            FillCsfRow(
                BandFrequency(band, bands, pixels_per_degree), channel,
                destination.subspan(row, kCsfSamples)
            );
        }
    }
}

CvvdpSpatial::CvvdpSpatial(VulkanRuntime& runtime) : runtime_(&runtime) {
    pipelines_ = runtime.CreatePipelines(
        std::array{"cvvdp_pyramid", "cvvdp_contrast", "cvvdp_mask",
                   "cvvdp_reduce"}
    );
}

CvvdpSpatial::~CvvdpSpatial() { runtime_->DestroyPipelines(pipelines_); }

void CvvdpSpatial::Record(
    Size size, double pixels_per_degree, std::uint32_t channels,
    std::uint32_t reference, std::uint32_t distorted, std::uint32_t scratch,
    std::uint32_t tables, std::uint32_t partials, std::uint32_t results
) {
    if (channels != 3 && channels != 4)
        throw std::invalid_argument(
            "cvvdp: spatial processing needs 3 or 4 channels"
        );
    SpatialLayout layout(size, pixels_per_degree, scratch);
    layout.levels[0].reference = reference;
    layout.levels[0].distorted = distorted;
    SpatialPush push{};
    push.channels = channels;
    push.bands = Word(layout.levels.size());
    push.tables = tables;
    push.partials = partials;
    push.results = results;
    RecordGaussianPyramid(*runtime_, pipelines_[kPyramid], push, layout);
    for (std::size_t band = 0; band < layout.levels.size(); ++band) {
        const auto& level = layout.levels[band];
        const bool baseband = band + 1 == layout.levels.size();
        if (baseband) {
            push.width = Word(level.size.width);
            push.height = Word(level.size.height);
            push.reference = level.reference;
            push.distorted = level.distorted;
            push.destination = layout.mutual;
            push.op = 0;
            runtime_->Dispatch(
                pipelines_[kReduce], &push, sizeof(push), 1, 2, 1
            );
            runtime_->ComputeBarrier();
        } else {
            RecordExpandedBand(
                *runtime_, pipelines_[kPyramid], push, layout, band
            );
        }
        push.width = Word(level.size.width);
        push.height = Word(level.size.height);
        push.reference = level.reference;
        push.distorted = level.distorted;
        push.destination = layout.temporary;
        push.auxiliary = layout.mutual;
        push.scratch = baseband ? layout.mutual : layout.expanded;
        push.band = Word(band);
        push.op = baseband ? 1u : 0u;
        const auto groups = Groups(Pixels(level.size), kLanes);
        runtime_->Dispatch(
            pipelines_[kContrast], &push, sizeof(push), groups, channels, 1
        );
        runtime_->ComputeBarrier();
        RecordMask(*runtime_, pipelines_[kMask], push, layout, baseband);
        push.op = 1;
        runtime_->Dispatch(
            pipelines_[kReduce], &push, sizeof(push), 1, channels, 1
        );
        runtime_->ComputeBarrier();
    }
}

}  // namespace vqm::internal
