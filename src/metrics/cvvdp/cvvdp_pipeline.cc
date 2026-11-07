// ColorVideoVDP model: Graphics and Displays Group (2024), MIT.
// VQM implementation: BSD-3-Clause; see LICENSE and licenses/.
#include "cvvdp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cvvdp_internal.h"
#include "cvvdp_spatial.h"
#include "handles.h"
#include "image_view.h"
#include "input.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_shared_storage.h"

namespace vqm {
namespace {
using namespace internal;

std::uint32_t Reserve(std::uint64_t& cursor, std::size_t words) {
    constexpr std::uint64_t limit = UINT32_MAX;
    if (words > limit - 15 ||
        cursor > limit - ((words + 15) & ~std::uint64_t{15}))
        throw std::invalid_argument("cvvdp: arena exceeds 32-bit addressing");
    const auto offset = static_cast<std::uint32_t>(cursor);
    cursor += (words + 15) & ~std::uint64_t{15};
    return offset;
}

// Only scratch and the current pair occupy the shared arena. Separate
// device-local histories avoid charging descriptor range for both inputs
// together. No borrowed buffer or host pointer becomes temporal history.
struct Layout {
    std::array<std::uint32_t, 2> source{}, desc{}, current{};
    std::vector<std::uint32_t> resident;
    std::uint32_t scratch, tables, filters, partials, results, status;
    std::size_t pixels, taps, levels, table_words, staging_tables, readback;
    StagingSources staging;
    BufferMemoryPlan memory;

    Layout(const CvvdpConfig& config, const VulkanContext& context) {
        ValidateMemoryPolicy(config.memory_policy);
        ValidateCvvdpDisplay(config.display);
        if (config.size.width == 0 || config.size.height == 0 ||
            config.size.width > UINT32_MAX || config.size.height > UINT32_MAX ||
            config.size.width > UINT32_MAX / config.size.height ||
            config.gpu_resident_images > kResidentIndexMask ||
            static_cast<unsigned>(config.chroma_filter) > 2)
            throw std::invalid_argument(
                "cvvdp: invalid geometry, resident count or chroma filter"
            );
        pixels = config.size.width * config.size.height;
        taps = config.temporal ? CvvdpTemporalTaps(config) : 0;
        const double ppd = config.display.pixels_per_degree;
        levels = CvvdpSpatial::Levels(config.size, ppd);
        table_words = CvvdpSpatial::TableWords(config.size, ppd);
        std::uint64_t cursor = 0;
        for (auto& offset : desc) offset = Reserve(cursor, kSourceDescWords);
        if (config.streaming_inputs)
            for (auto& offset : source) offset = Reserve(cursor, 4 * pixels);
        for (std::size_t i = 0; i < config.gpu_resident_images; ++i)
            resident.push_back(Reserve(cursor, 4 * pixels));
        for (auto& offset : current) offset = Reserve(cursor, 4 * pixels);
        scratch = Reserve(cursor, CvvdpSpatial::ScratchWords(config.size, ppd));
        tables = Reserve(cursor, table_words);
        filters = Reserve(cursor, 4 * taps);
        partials = Reserve(cursor, 4 * CvvdpSpatial::PartialCount(config.size));
        results = Reserve(cursor, 4 * levels + 1);
        status = results + static_cast<std::uint32_t>(4 * levels);
        memory.arena_bytes = Bytes(cursor);
        memory.persistent_ranges = {{Bytes(tables), Bytes(table_words), true}};
        if (taps != 0)
            memory.persistent_ranges.push_back(
                {Bytes(filters), Bytes(4 * taps), true}
            );
        staging = PlanStagingSources(
            pixels, config.streaming_inputs, config.gpu_resident_images
        );
        staging_tables = staging.reserved_floats + 2 * kSourceDescWords;
        readback = staging_tables + table_words + 4 * taps;
        memory.fallback_staging_bytes = Bytes(readback + 4 * levels + 1);
        if (taps != 0)
            memory.auxiliary_bytes.assign(2, Bytes(3 * pixels * taps));
        const auto& limits = context.limits;
        const auto max_groups = limits.maxComputeWorkGroupCount[0];
        if (limits.maxComputeWorkGroupInvocations < 256 ||
            limits.maxComputeWorkGroupSize[0] < 256 ||
            limits.maxComputeWorkGroupSize[1] < 8 ||
            limits.maxComputeWorkGroupSize[2] < 1 ||
            limits.maxComputeWorkGroupCount[2] < 1 ||
            limits.maxComputeWorkGroupCount[1] < 8 ||
            limits.maxComputeSharedMemorySize < 4096 ||
            limits.maxPushConstantsSize < 128 ||
            limits.maxPerStageDescriptorStorageBuffers < (taps ? 6u : 4u) ||
            limits.maxDescriptorSetStorageBuffers < (taps ? 6u : 4u) ||
            Groups(config.size.width, 32) > max_groups ||
            Groups(config.size.height, 16) >
                limits.maxComputeWorkGroupCount[1] ||
            Groups(pixels, 256) > max_groups)
            throw std::invalid_argument(
                "cvvdp: compute dispatch exceeds device limits"
            );
        if (memory.arena_bytes > limits.maxStorageBufferRange ||
            std::any_of(
                memory.auxiliary_bytes.begin(), memory.auxiliary_bytes.end(),
                [&](auto bytes) {
                    return bytes > limits.maxStorageBufferRange ||
                           bytes > Bytes(UINT32_MAX);
                }
            ))
            throw std::invalid_argument(
                "cvvdp: geometry/frame rate exceeds device storage range"
            );
    }
};

struct TemporalPush {
    std::uint32_t pixels, taps, current, available, filters, reference,
        distorted;
};
using Sources = std::array<VulkanRuntime::Source, 2>;
}  // namespace

namespace internal {
BufferMemoryPlan PlanCvvdpMemory(
    const CvvdpConfig& config, const VulkanContext& context
) {
    return Layout(config, context).memory;
}
}  // namespace internal

struct CvvdpPipeline::Impl {
    Impl(
        CvvdpConfig c, const VulkanDevice* device,
        std::shared_ptr<SharedMetricStorage> shared = {}
    )
        : config(std::move(c)),
          context(
              shared ? shared->backing().shared_context()
                     : RequireVulkanContext(device)
          ),
          layout(config, *context),
          runtime(context, RuntimeConfig(), std::move(shared)) {
        runtime.CreateArena(layout.memory.arena_bytes);
        runtime.CreateAuxiliaryStorage();
        if (!runtime.arena_visible())
            runtime.CreateStaging(layout.memory.fallback_staging_bytes);
        runtime.CreateCommandObjects();
        runtime.CreateDescriptors(0, layout.memory.arena_bytes);
        prepare.emplace(runtime, config.display, config.chroma_filter);
        spatial.emplace(runtime);
        resident.Resize(config.gpu_resident_images);
        InitializeTables();
        runtime.InitializeSharedState();
        // Raw pipeline ownership starts after table publication and host
        // allocations, so constructor failure cannot leak this handle.
        if (config.temporal) temporal = runtime.CreatePipeline("cvvdp_temporal");
    }
    ~Impl() { runtime.DestroyPipelines(std::array{temporal}); }

    VulkanRuntime::Config RuntimeConfig() const {
        VulkanRuntime::Config out;
        out.stream_region = layout.source;
        out.stream_desc = layout.desc;
        out.resident_slots = layout.resident;
        out.staging_desc_offset = layout.staging.reserved_floats;
        out.staging_image_bytes = layout.staging.image_bytes;
        out.base_pixels = layout.pixels;
        out.base_size = config.size;
        out.streaming = config.streaming_inputs;
        out.memory_policy = config.memory_policy;
        out.chroma_filter = config.chroma_filter;
        out.push_constant_size = 128;
        out.persistent_ranges = layout.memory.persistent_ranges;
        out.auxiliary_bytes.assign(
            layout.memory.auxiliary_bytes.begin(),
            layout.memory.auxiliary_bytes.end()
        );
        return out;
    }
    void InitializeTables() {
        std::vector<float> values(layout.table_words + 4 * layout.taps);
        CvvdpSpatial::FillTables(
            config.size, config.display.pixels_per_degree,
            std::span(values).first(layout.table_words)
        );
        if (layout.taps) {
            const auto filters = CvvdpTemporalFilters(config);
            std::copy(
                filters.begin(), filters.end(),
                values.data() + layout.table_words
            );
        }
        // Arena alignment may separate the spatial and temporal tables.
        if (runtime.arena_visible()) {
            std::copy_n(
                values.data(), layout.table_words,
                runtime.arena_map() + layout.tables
            );
            std::copy(
                values.data() + layout.table_words,
                values.data() + values.size(),
                runtime.arena_map() + layout.filters
            );
        } else {
            std::copy(
                values.begin(), values.end(),
                runtime.staging_map() + layout.staging_tables
            );
            const std::array copies{
                VulkanRuntime::CopyRange{layout.staging_tables, layout.tables,
                                         layout.table_words},
                VulkanRuntime::CopyRange{
                    layout.staging_tables + layout.table_words, layout.filters,
                    4 * layout.taps}};
            runtime.BeginCommandBuffer();
            runtime.RecordStagedCopies(
                std::span(copies).first(layout.taps ? 2 : 1)
            );
            runtime.EndCommandBuffer();
            runtime.SubmitAndWait();
        }
    }
    void RequireOpen() const {
        if (failed)
            throw std::logic_error("cvvdp: processing failed; call Reset");
        if (finished)
            throw std::logic_error("cvvdp: clip finalized; call Reset");
    }
    void RequireImage() const {
        RequireOpen();
        if (frames != 0)
            throw std::logic_error(
                "cvvdp: reset active clip before ComputeImage"
            );
    }
    void ValidateFormat(const ImageFormat& format) const {
        ValidateCvvdpInput(format, config.display);
    }
    template <class Input>
    Sources Resolve(const Input& a, const Input& b) {
        ValidateFormat(a.format);
        ValidateFormat(b.format);
        runtime.UploadStreaming(a, b);
        return {};
    }
    Sources Resolve(const UploadedImage& a, const UploadedImage& b) {
        const auto& ref = resident.Require(a, config.size, "original");
        const auto& dist = resident.Require(b, config.size, "distorted");
        return {VulkanRuntime::Source{VulkanRuntime::Source::Kind::Resident,
                                      ref.device},
                VulkanRuntime::Source{VulkanRuntime::Source::Kind::Resident,
                                      dist.device}};
    }
    Sources Resolve(const VulkanBufferView& a, const VulkanBufferView& b) {
        ValidateFormat(a.format);
        ValidateFormat(b.format);
        return runtime.ResolveExternalInputs(a, b);
    }

    // Compute->transfer protects the converted current pair; transfer->compute
    // publishes the ring slot before FIR reads it. Both buffers remain owned.
    void StoreHistory() {
        runtime.Barrier(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
        );
        const auto slot = frames % layout.taps;
        for (std::size_t side = 0; side < 2; ++side) {
            const VkBufferCopy copy{Bytes(layout.current[side]),
                                    Bytes(slot * 3 * layout.pixels),
                                    Bytes(3 * layout.pixels)};
            vkCmdCopyBuffer(
                runtime.command_buffer(), runtime.arena_buffer(),
                runtime.auxiliary_buffer(side), 1, &copy
            );
        }
        runtime.Barrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        );
    }
    void RestoreFirstVideoFrame() {
        runtime.Barrier(
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
        );
        for (std::size_t side = 0; side < 2; ++side) {
            const VkBufferCopy copy{0, Bytes(layout.current[side]),
                                    Bytes(3 * layout.pixels)};
            vkCmdCopyBuffer(
                runtime.command_buffer(), runtime.auxiliary_buffer(side),
                runtime.arena_buffer(), 1, &copy
            );
        }
        runtime.Barrier(
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
        );
    }
    void FilterHistory(std::uint64_t index) {
        runtime.OpenStage(Stage::CvvdpTemporal);
        const TemporalPush push{
            static_cast<std::uint32_t>(layout.pixels),
            static_cast<std::uint32_t>(layout.taps),
            static_cast<std::uint32_t>(index % layout.taps),
            static_cast<std::uint32_t>(
                std::min<std::uint64_t>(index + 1, layout.taps)
            ),
            layout.filters,
            layout.current[0],
            layout.current[1]};
        runtime.Dispatch(
            temporal, &push, sizeof(push), Groups(layout.pixels, 256), 4, 1
        );
        runtime.ComputeBarrier();
        runtime.CloseStage();
    }
    void RecordSpatial(std::uint32_t channels) {
        runtime.OpenStage(Stage::CvvdpSpatial);
        spatial->Record(
            config.size, config.display.pixels_per_degree, channels,
            layout.current[0], layout.current[1], layout.scratch, layout.tables,
            layout.partials, layout.results
        );
        runtime.CloseStage();
    }
    void RecordReadback() {
        if (runtime.arena_visible())
            runtime.ComputeToHostBarrier();
        else
            runtime.RecordReadback(
                layout.results, layout.readback, 4 * layout.levels + 1
            );
    }
    void CheckInputStatus() const {
        std::uint32_t status = 0;
        const auto* values =
            runtime.arena_visible()
                ? runtime.arena_map() + layout.status
                : runtime.staging_map() + layout.readback + 4 * layout.levels;
        std::memcpy(&status, values, sizeof(status));
        if (status != 0)
            throw std::invalid_argument(
                "cvvdp: input contains non-finite samples"
            );
    }
    std::span<const float> Results(std::size_t channels) const {
        return {runtime.arena_visible()
                    ? runtime.arena_map() + layout.results
                    : runtime.staging_map() + layout.readback,
                channels * layout.levels};
    }
    void Finish(std::uint64_t token, const VulkanSubmitSync* sync) {
        runtime.EndCommandBuffer();
        if (sync)
            runtime.QueueSubmit(*sync);
        else
            runtime.QueueSubmit();
        runtime.FinishSubmission(token);
    }
    void AccumulateFrame() {
        VQM_TIMED(CvvdpScore);
        const double distortion = CvvdpFrameDistortion(Results(4), 4);
        power_sum += CvvdpSafePow(distortion, 2.0);
        if (frame_scores_enabled)
            frame_scores[frame_score_count++] = CvvdpJod(distortion);
    }
    void ProcessFirstVideoFrame() {
        const auto token = runtime.BeginSubmission();
        runtime.BeginCommandBuffer();
        runtime.BindDescriptorSet();
        runtime.HostToComputeBarrier();
        FilterHistory(0);
        RecordSpatial(4);
        RecordReadback();
        Finish(token, nullptr);
        AccumulateFrame();
    }
    template <class Input>
    CvvdpScore Run(
        const Input& a, const Input& b, bool image,
        const VulkanSubmitSync* sync = nullptr
    ) {
        if (image)
            RequireImage();
        else {
            RequireOpen();
            if (!config.temporal)
                throw std::logic_error("cvvdp: temporal storage disabled");
            if (frames == UINT64_MAX)
                throw std::overflow_error("cvvdp: too many frames");
        }
        if (sync) VulkanRuntime::ValidateSubmitSync(*sync);
        const auto sources =
            Resolve(a, b);  // Invalid inputs do not advance the clip.
        try {
            if (!image && frame_scores_enabled) frame_score_count = 0;
            if (!image && frames == 1) ProcessFirstVideoFrame();
            runtime.PrepareSourceBindings(sources);
            const auto token = runtime.BeginSubmission();
            runtime.BeginCommandBuffer();
            const auto formats = runtime.PublishSources(sources);
            runtime.FillZero(layout.status, 1);
            runtime.Barrier(
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
            );
            runtime.OpenStage(Stage::CvvdpPrepare);
            for (std::size_t side = 0; side < 2; ++side)
                prepare->Record(
                    side, layout.desc[side], layout.current[side], config.size,
                    *formats[side], layout.status
                );
            runtime.ComputeBarrier();
            runtime.CloseStage();
            if (!image) StoreHistory();
            if (!image && frames != 0) FilterHistory(frames);
            if (image || frames != 0) RecordSpatial(image ? 3 : 4);
            RecordReadback();
            Finish(token, sync);
            CheckInputStatus();
            if (image) {
                VQM_TIMED(CvvdpScore);
                return {CvvdpImageJod(Results(3)), 1};
            }
            if (frames != 0) AccumulateFrame();
            ++frames;
            return {};
        } catch (...) {
            frame_score_count = 0;
            failed = true;
            throw;
        }
    }
    std::optional<CvvdpScore> Finalize() {
        if (failed)
            throw std::logic_error("cvvdp: processing failed; call Reset");
        if (finished) return final_score;
        try {
            if (frames == 1) {
                const auto token = runtime.BeginSubmission();
                runtime.BeginCommandBuffer();
                runtime.BindDescriptorSet();
                runtime.HostToComputeBarrier();
                RestoreFirstVideoFrame();
                RecordSpatial(3);
                RecordReadback();
                Finish(token, nullptr);
                final_score = CvvdpScore{CvvdpImageJod(Results(3)), frames};
            } else if (frames != 0) {
                final_score = CvvdpScore{
                    CvvdpJod(
                        CvvdpSafePow(power_sum / static_cast<double>(frames), .5)
                    ),
                    frames};
            }
            finished = true;
            return final_score;
        } catch (...) {
            failed = true;
            throw;
        }
    }
    void Reset() {
        // A rejected recording can leave the runtime token pending without a
        // submission. Destruction/reconstruction is required for device errors;
        // ordinary numerical/processing errors finish their submission first.
        if (runtime.pending())
            throw std::logic_error(
                "cvvdp: backend submission failed; reconstruct pipeline"
            );
        frames = 0;
        power_sum = 0;
        frame_score_count = 0;
        failed = false;
        finished = false;
        final_score.reset();
    }

    CvvdpConfig config;
    std::shared_ptr<VulkanContext> context;
    Layout layout;
    VulkanRuntime runtime;
    std::optional<CvvdpPrepare> prepare;
    std::optional<CvvdpSpatial> spatial;
    VkPipeline temporal = VK_NULL_HANDLE;
    ResidentTable resident;
    std::uint64_t frames = 0;
    double power_sum = 0;
    std::array<double, 2> frame_scores{};
    std::size_t frame_score_count = 0;
    bool frame_scores_enabled = false;
    bool failed = false, finished = false;
    std::optional<CvvdpScore> final_score;
};

CvvdpPipeline::CvvdpPipeline(CvvdpConfig config)
    : impl_(std::make_unique<Impl>(std::move(config), nullptr)) {}
CvvdpPipeline::CvvdpPipeline(CvvdpConfig config, const VulkanDevice& device)
    : impl_(std::make_unique<Impl>(std::move(config), &device)) {}
CvvdpPipeline::~CvvdpPipeline() = default;
CvvdpPipeline::CvvdpPipeline(
    CvvdpConfig config, const VulkanDevice* device,
    std::shared_ptr<internal::SharedMetricStorage> scratch
)
    : impl_(std::make_unique<Impl>(config, device, std::move(scratch))) {}
CvvdpPipeline::CvvdpPipeline(CvvdpPipeline&&) noexcept = default;
CvvdpPipeline& CvvdpPipeline::operator=(CvvdpPipeline&&) noexcept = default;
UploadedImage CvvdpPipeline::Upload(const Image& image) {
    impl_->runtime.RequireIdle();
    impl_->ValidateFormat(image.format);
    CheckImage(image, impl_->config.size, "image");
    return impl_->resident.Upload(image.size, [&](std::size_t count) {
        return impl_->runtime.UploadResident(image, count);
    });
}
void CvvdpPipeline::Release(UploadedImage image) {
    impl_->runtime.RequireIdle();
    impl_->resident.Release(image, [&](auto slot) {
        impl_->runtime.ReleaseResident(slot);
    });
}
CvvdpScore CvvdpPipeline::ComputeImage(const Image& a, const Image& b) {
    return impl_->Run(a, b, true);
}
CvvdpScore CvvdpPipeline::ComputeImage(const ImageView& a, const ImageView& b) {
    return impl_->Run(a, b, true);
}
CvvdpScore CvvdpPipeline::ComputeImage(
    const UploadedImage& a, const UploadedImage& b
) {
    return impl_->Run(a, b, true);
}
CvvdpScore CvvdpPipeline::ComputeImage(
    const VulkanBufferView& a, const VulkanBufferView& b
) {
    return impl_->Run(a, b, true);
}
CvvdpScore CvvdpPipeline::ComputeImage(
    const VulkanBufferView& a, const VulkanBufferView& b,
    const VulkanSubmitSync& sync
) {
    return impl_->Run(a, b, true, &sync);
}
void CvvdpPipeline::Push(const Image& a, const Image& b) {
    (void)impl_->Run(a, b, false);
}
void CvvdpPipeline::Push(const ImageView& a, const ImageView& b) {
    (void)impl_->Run(a, b, false);
}
void CvvdpPipeline::Push(const UploadedImage& a, const UploadedImage& b) {
    (void)impl_->Run(a, b, false);
}
void CvvdpPipeline::Push(const VulkanBufferView& a, const VulkanBufferView& b) {
    (void)impl_->Run(a, b, false);
}
void CvvdpPipeline::Push(
    const VulkanBufferView& a, const VulkanBufferView& b,
    const VulkanSubmitSync& sync
) {
    (void)impl_->Run(a, b, false, &sync);
}
std::optional<CvvdpScore> CvvdpPipeline::Finalize() { return impl_->Finalize(); }
void CvvdpPipeline::Reset() { impl_->Reset(); }
PipelineMemoryUsage CvvdpPipeline::GetMemoryUsage() const {
    return impl_->runtime.GetMemoryUsage();
}

void internal::CvvdpFrameAccess::Enable(CvvdpPipeline& pipeline) {
    pipeline.impl_->frame_scores_enabled = true;
}

std::span<const double> internal::CvvdpFrameAccess::Scores(
    const CvvdpPipeline& pipeline
) {
    const auto& impl = *pipeline.impl_;
    return {impl.frame_scores.data(), impl.frame_score_count};
}
}  // namespace vqm
