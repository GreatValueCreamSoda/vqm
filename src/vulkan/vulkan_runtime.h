// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// The Vulkan plumbing every metric backend shares: one arena and one staging
// buffer, the descriptor scheme (arena floats, plane region, arena words and
// source words), the single command buffer and fence with its one-in-flight
// token, dispatch and barrier wrappers, and the four input paths: streamed
// host pixels, resident slots, allocated image buffers and borrowed Vulkan
// buffers. It also records staged copies, readback and opt-in GPU stage timers.
//
// A backend supplies what is metric-specific: its arena layout and the offsets
// the input machinery needs (Config), its push-constant block, its pipelines
// and its recording order. Every metric shares the input, submission and
// synchronization machinery; the runtime does not interpret metric features.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <vulkan/vulkan.h>

#include "timing.h"
#include "vqm_image.h"
#include "vqm_memory_types.h"
#include "vqm_vulkan.h"
#include "vulkan_support.h"

namespace vqm::internal {

// Reject invalid public policy values consistently in planning and creation.
void ValidateMemoryPolicy(MemoryPolicy policy);

class SharedMetricStorage;

// Throws std::runtime_error naming `what` unless `result` is VK_SUCCESS.
void Check(VkResult result, const char* what);

// Byte size of a float count in the arena.
[[nodiscard]] VkDeviceSize Bytes(std::size_t floats);

// Bytes per plane element: four in exact mode, two when the planes are stored
// as 16-bit floats.
[[nodiscard]] VkDeviceSize PlaneBytes(std::uint32_t elements, bool half);

// Workgroups needed for `extent` at `per_group`, at least one.
[[nodiscard]] std::uint32_t Groups(std::size_t extent, std::uint32_t per_group);

// Floats of arena one input image needs: four planes of at most four bytes per
// sample, 16 bytes per pixel. The staging upload area holds two of these.
inline constexpr std::size_t kSourceFloatsPerPixel = 4;

struct StagingSources {
    std::size_t reserved_floats;
    std::size_t image_bytes;
};

// Reserve two image regions for streaming, one for resident uploads, and none
// when neither input path is configured. The rest of each staging layout
// starts after this region, so resident uploads cannot overwrite its tables.
[[nodiscard]] constexpr StagingSources PlanStagingSources(
    std::size_t pixels, bool streaming, std::size_t resident_images
) {
    const std::size_t image_floats = kSourceFloatsPerPixel * pixels;
    return {
        streaming ? 2 * image_floats : (resident_images != 0 ? image_floats : 0),
        (streaming || resident_images != 0) ? image_floats * sizeof(float) : 0};
}

class VulkanRuntime {
public:
    struct Allocation {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize allocation_bytes = 0;
        std::uint32_t memory_type = UINT32_MAX;
        std::uint32_t heap_index = UINT32_MAX;
    };

    // Where one side of the pair comes from.
    struct Source {
        enum class Kind : std::uint8_t {
            Stream,       // copied into the arena's source region
            Resident,     // an Upload() slot, already in the arena
            ImageBuffer,  // a caller-owned carve-out of the input pool
            External,     // borrowed VkBuffer; read directly by the shader
        };
        Kind kind = Kind::Stream;
        std::uint32_t index = 0;  // resident slot or input_allocations index
        const VulkanBufferView* external = nullptr;
    };

    // One caller-owned carve-out: RequiredBytes(format) bytes, 64-byte aligned,
    // inside the arena's input pool.
    struct InputAllocation {
        std::uint32_t offset = 0;  // arena floats
        std::uint32_t floats = 0;
        std::size_t bytes = 0;
        ImageFormat format{};
        std::uint64_t generation = 0;
        bool live = false;
    };

    // One staged copy: `floats` from the staging buffer at `staging_floats` to
    // the arena at `arena_floats`. Both count floats, which are words.
    struct CopyRange {
        std::size_t staging_floats = 0;
        std::uint32_t arena_floats = 0;
        std::size_t floats = 0;
    };

    // The arena offsets the input machinery needs. Everything is the backend's
    // layout, in arena floats except `staging_desc_offset`, which is a staging
    // float offset.
    struct Config {
        std::array<std::uint32_t, 2> stream_region{};  // arena floats
        std::array<std::uint32_t, 2> stream_desc{};    // arena floats
        std::vector<std::uint32_t> resident_slots;     // arena floats
        std::uint32_t input_pool = 0;                  // arena floats
        std::uint32_t input_pool_size = 0;             // arena floats
        std::size_t staging_desc_offset = 0;           // staging floats
        // Bytes of staging word 0 reserved for one raw image, which is where
        // UploadResident stages a resident upload on a device whose arena is
        // not host visible. Zero when the pipeline reserves no such area. The
        // descriptors and level tables live after it, so an upload may not
        // write past this many bytes.
        std::size_t staging_image_bytes = 0;
        std::size_t base_pixels = 0;
        Size base_size{};
        bool streaming = true;
        MemoryPolicy memory_policy = MemoryPolicy::Automatic;
        ChromaFilter chroma_filter = ChromaFilter::Bilinear;
        std::size_t push_constant_size = 0;
        // Optional device-local storage bindings 4.., owned by this runtime.
        // Existing metrics keep the original four-binding descriptor layout.
        std::vector<VkDeviceSize> auxiliary_bytes;
    };

    struct SourceCopies {
        std::array<CopyRange, 4> ranges{};
        std::size_t count = 0;
    };

    VulkanRuntime(
        std::shared_ptr<VulkanContext> context, Config config,
        std::shared_ptr<SharedMetricStorage> shared = {}
    );
    ~VulkanRuntime();
    VulkanRuntime(const VulkanRuntime&) = delete;
    VulkanRuntime& operator=(const VulkanRuntime&) = delete;

    // Resources, in this order: the arena decides whether staging exists at
    // all, and the descriptors point at both.
    void CreateArena(VkDeviceSize bytes);
    void CreateAuxiliaryStorage();
    void CreateStaging(std::size_t bytes);
    void CreateCommandObjects();
    void CreateDescriptors(
        VkDeviceSize plane_base_bytes, VkDeviceSize plane_bytes
    );
    void Destroy() noexcept;

    [[nodiscard]] VulkanContext& context() const noexcept { return *ctx_; }
    [[nodiscard]] bool arena_visible() const noexcept { return arena_visible_; }
    [[nodiscard]] PipelineMemoryUsage GetMemoryUsage() const;
    [[nodiscard]] const Allocation& arena_allocation() const noexcept {
        return arena_;
    }
    [[nodiscard]] const Allocation& staging_allocation() const noexcept {
        return staging_;
    }
    [[nodiscard]] VkDeviceSize arena_bytes() const noexcept {
        return arena_bytes_;
    }
    [[nodiscard]] VkDeviceSize staging_bytes() const noexcept {
        return staging_bytes_;
    }
    [[nodiscard]] float* arena_map() const noexcept { return arena_map_; }
    [[nodiscard]] VkBuffer arena_buffer() const noexcept {
        return arena_.buffer;
    }
    [[nodiscard]] VkBuffer auxiliary_buffer(std::size_t index) const {
        return auxiliary_.at(index).buffer;
    }
    [[nodiscard]] float* staging_map() const noexcept { return staging_map_; }
    [[nodiscard]] VkCommandBuffer command_buffer() const noexcept {
        return command_buffer_;
    }
    [[nodiscard]] VkPipelineLayout pipeline_layout() const noexcept {
        return pipeline_layout_;
    }
    [[nodiscard]] VkDescriptorSet descriptor_set() const noexcept {
        return descriptor_sets_[0];
    }

    // Builds a compute pipeline from the embedded blob named `name`; the
    // shader module lives only for the call.
    [[nodiscard]] VkPipeline CreatePipeline(std::string_view name);

    // Creates an ordered set transactionally: failure destroys earlier handles.
    // The caller owns the returned handles and must destroy them before this
    // runtime, after all submissions that use them have completed.
    template <std::size_t N>
    [[nodiscard]] std::array<VkPipeline, N> CreatePipelines(
        const std::array<const char*, N>& names
    ) {
        std::array<VkPipeline, N> pipelines{};
        try {
            for (std::size_t i = 0; i < names.size(); ++i)
                pipelines[i] = CreatePipeline(names[i]);
        } catch (...) {
            DestroyPipelines(pipelines);
            throw;
        }
        return pipelines;
    }
    void DestroyPipelines(std::span<const VkPipeline> pipelines) noexcept;

    // Recording. Begin/End wrap one submission and reset the stage-query pool.
    // BindSource or BindDescriptorSet establishes bindings before dispatch.
    void BeginCommandBuffer();
    void EndCommandBuffer();
    void ResetCommandPool();
    void BindDescriptorSet();
    // Selects the source binding for this image's prepare dispatch.
    void BindSource(std::size_t image);
    void Dispatch(
        VkPipeline pipeline, const void* params, std::size_t params_bytes,
        std::uint32_t groups_x, std::uint32_t groups_y, std::uint32_t groups_z
    );
    void Barrier(
        VkPipelineStageFlags src_stage, VkAccessFlags src_access,
        VkPipelineStageFlags dst_stage, VkAccessFlags dst_access
    );
    void ComputeBarrier();
    void HostToComputeBarrier();
    void ComputeToHostBarrier();
    void CopyUpload(
        std::size_t staging_floats, std::uint32_t arena_floats,
        std::size_t floats
    );
    // Zeros `floats` arena floats at `arena_floats` (vkCmdFillBuffer, 0 bits is
    // +0.0f). Used for accumulators that the shaders only add into.
    void FillZero(std::uint32_t arena_floats, std::size_t floats);
    // host->transfer, the copies, transfer->compute. No begin/end/submit: the
    // caller decides whether these join a pair's command buffer or a one-shot
    // upload submission.
    void RecordStagedCopies(std::span<const CopyRange> copies);
    // compute->transfer, one copy out, transfer->host.
    void RecordReadback(
        std::uint32_t arena_floats, std::size_t staging_floats,
        std::size_t floats
    );
    // The whole one-shot copy: BeginCommandBuffer, the ranges, End, submit and
    // wait. The caller has already filled the staging buffer.
    void PublishStaged(
        std::uint32_t arena_floats, std::size_t staging_floats,
        std::size_t floats
    );

    // Submission. One in flight per runtime: BeginSubmission throws
    // std::logic_error while one is pending and returns the token that
    // FinishSubmission consumes.
    void RequireIdle() const;
    [[nodiscard]] std::uint64_t BeginSubmission();
    void QueueSubmit();
    // Validate binary semaphore lists while idle, before recording any work.
    static void ValidateSubmitSync(const VulkanSubmitSync& sync);
    void QueueSubmit(const VulkanSubmitSync& sync);
    void WaitFence();
    void SubmitAndWait();  // QueueSubmit + WaitFence, no token involved
    [[nodiscard]] bool pending() const noexcept { return pending_; }
    [[nodiscard]] std::uint64_t token() const noexcept { return token_; }
    // Waits, clears the in-flight flag and collects the stage timings. Throws
    // std::invalid_argument for a token that is not the one in flight.
    void FinishSubmission(std::uint64_t token);

    // Inputs. The streamed path writes the raw pixels and the descriptors the
    // host side built; the resident and carve-out paths are already in the
    // arena and only need descriptors.
    void UploadStreaming(const Image& original, const Image& distorted);
    void UploadStreaming(const ImageView& original, const ImageView& distorted);
    void UploadStreaming(
        const Image& original, const Image& distorted,
        std::array<std::uint8_t, 2> plane_masks
    );
    void UploadStreaming(
        const ImageView& original, const ImageView& distorted,
        std::array<std::uint8_t, 2> plane_masks
    );
    void UploadStreamingDistorted(const Image& distorted);
    // Validate source descriptions before any command buffer is recorded.
    // The returned views borrow the caller's VulkanBufferView objects.
    [[nodiscard]] std::array<Source, 2> ResolveExternalInputs(
        const VulkanBufferView& original, const VulkanBufferView& distorted
    ) const;
    // Update source bindings while idle, before command buffer recording.
    // Internal inputs bind the arena, even after an external pair was used.
    void PrepareSourceBindings(const std::array<Source, 2>& sources);
    // Resolve and publish the descriptors and any staged pixels to the arena.
    // The caller has begun recording and controls the stage timing boundary.
    [[nodiscard]] std::array<const ImageFormat*, 2> PublishSources(
        const std::array<Source, 2>& sources
    );
    // Build the staging copies independently of the device, so the fallback
    // path can be checked even on a host-visible arena.
    [[nodiscard]] static SourceCopies PlanSourceCopies(
        const Config& config, const std::array<Source, 2>& sources,
        const std::array<const ImageFormat*, 2>& formats,
        std::array<std::uint8_t, 2> plane_masks = {0x0f, 0x0f}
    );
    std::optional<std::uint32_t> UploadResident(
        const Image& image, std::size_t capacity
    );
    void ReleaseResident(std::uint32_t slot);

    // Caller-owned input pool.
    [[nodiscard]] ImageBuffer AllocateImage(const ImageFormat& format);
    [[nodiscard]] std::byte* MapImage(ImageBuffer buffer);
    void ReleaseInput(ImageBuffer buffer);
    // Validate both buffers and map them to arena sources before recording.
    [[nodiscard]] std::array<Source, 2> ResolveInputs(
        const ImageBuffer& original, const ImageBuffer& distorted
    ) const;

    // Stage timing; every one is a no-op without VQM_STAGE_TIMING.
    void ResetTiming();
    void OpenStage(Stage stage);
    void CloseStage();
    void CollectTimings();

private:
    friend class SharedMetricStorage;
    void QueueSubmitInfo(const VkSubmitInfo& submit);
    [[nodiscard]] std::byte* StreamingDestination(std::size_t image) const;
    void WriteStreamingSource(
        std::size_t image, const Image& source, std::uint8_t plane_mask
    );
    void WriteSourceDesc(
        std::size_t image, const ImageFormat& format,
        std::uint32_t plane_base_word, const VulkanBufferView* external
    );
    struct ResolvedSource {
        const ImageFormat* format;
        std::uint32_t plane_base;
    };
    [[nodiscard]] ResolvedSource ResolveSource(
        const Source& source, std::size_t image
    ) const;
    [[nodiscard]] VkDescriptorBufferInfo ExternalBinding(
        const VulkanBufferView& image, std::uint32_t* plane_base
    ) const;
    [[nodiscard]] std::size_t staging_source_offset(std::size_t image) const {
        return image * kSourceFloatsPerPixel * config_.base_pixels;
    }
    [[nodiscard]] std::optional<std::size_t> ResolveInput(
        ImageBuffer buffer
    ) const;
    Allocation CreateAllocation(
        VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties
    );
    [[nodiscard]] std::uint32_t FindMemoryType(
        std::uint32_t type_bits, VkMemoryPropertyFlags required
    ) const;
    [[nodiscard]] std::optional<std::uint32_t> FindMemoryTypeIfSupported(
        std::uint32_t type_bits, VkMemoryPropertyFlags required
    ) const;

#ifdef VQM_STAGE_TIMING
    struct TimedRange {
        Stage stage;
        std::uint32_t begin;
        std::uint32_t end;
    };
#endif

    std::shared_ptr<VulkanContext> ctx_;
    std::shared_ptr<SharedMetricStorage> shared_;
    Config config_;

    Allocation arena_;
    VkDeviceSize arena_bytes_ = 0;
    // Device-local memory the host can write directly (resizable BAR); when
    // present the upload skips both the staging fill and the DMA copy.
    float* arena_map_ = nullptr;
    bool arena_visible_ = false;
    Allocation staging_;
    std::vector<Allocation> auxiliary_;
    VkDeviceSize staging_bytes_ = 0;
    float* staging_map_ = nullptr;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    // One submission at a time: Submit queues the pair and returns a token,
    // Finish waits on it. Concurrency comes from holding several runtimes.
    bool pending_ = false;
    std::uint64_t token_ = 0;

    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> descriptor_sets_{};
    std::array<VkDescriptorBufferInfo, 2> source_bindings_{};

    // The format each buffer holds: the streamed source regions, and every
    // resident slot. The descriptor is rebuilt from these on every record.
    std::array<ImageFormat, 2> stream_format_{};
    std::array<std::uint8_t, 2> stream_plane_masks_{0x0f, 0x0f};
    std::vector<ImageFormat> resident_format_;
    // One entry per arena resident slot; true while the slot is free.
    std::vector<bool> resident_free_;

    std::vector<InputAllocation> input_allocations_;
    std::uint32_t input_pool_ = 0;       // arena floats
    std::uint32_t input_pool_size_ = 0;  // arena floats

#ifdef VQM_STAGE_TIMING
    VkQueryPool queries_ = VK_NULL_HANDLE;
    std::vector<TimedRange> timed_ranges_;
    std::uint32_t next_timestamp_ = 0;
    std::int32_t open_timestamp_ = -1;
    Stage open_stage_ = Stage::Ssimulacra2Prepare;
#endif
};

}  // namespace vqm::internal
