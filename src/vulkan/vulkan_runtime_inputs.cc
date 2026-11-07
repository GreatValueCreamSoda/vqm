// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vulkan_runtime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "handles.h"
#include "image_view.h"
#include "input.h"

namespace vqm::internal {
namespace {

// The caller-owned input pool packs a carve-out index with its generation the
// same way UploadedImage does for resident slots.
constexpr std::uint32_t kInputIndexBits = 12;
constexpr std::uint32_t kInputIndexMask = (1u << kInputIndexBits) - 1u;
static_assert(kInputIndexBits == kResidentIndexBits);

struct ExternalPlanes {
    std::array<VkDeviceSize, 4> offsets{};
    std::array<std::uint32_t, 4> row_samples{};
    VkDeviceSize required = 0;
};

// Find the first aligned gap in the live input spans. Offsets and lengths are
// in float words relative to the input pool; the spans must be sorted by
// offset. Only starts are aligned, so an exactly sized pool can hold one image.
std::optional<std::uint64_t> FindInputSpan(
    std::span<const std::pair<std::uint32_t, std::uint32_t>> used,
    std::uint64_t words, std::uint64_t pool_size
) {
    const auto align = [](std::uint64_t offset) {
        return (offset + 15u) / 16u * 16u;
    };
    std::uint64_t cursor = 0;
    for (const auto& [offset, length] : used) {
        const std::uint64_t start = align(cursor);
        if (start <= offset && words <= offset - start) return start;
        cursor = std::max(cursor, std::uint64_t{offset} + length);
    }
    const std::uint64_t start = align(cursor);
    if (start <= pool_size && words <= pool_size - start) return start;
    return std::nullopt;
}

// Bytes from the start of a packed image through the last selected plane.
// Callers use a prefix copy so a luma-only mask does not stage chroma.
std::size_t SelectedPlaneBytes(
    const ImageFormat& format, Size size, std::uint8_t plane_mask
) {
    const auto layout = PlaneLayout(format, size);
    std::size_t end = 0;
    for (std::size_t plane = 0; plane < layout.size(); ++plane) {
        const PlaneGeometry& geometry = layout[plane];
        if (!geometry.present() || (plane_mask & (1u << plane)) == 0) continue;
        end = std::max(
            end, geometry.offset +
                     geometry.width * geometry.height * geometry.sample_bytes
        );
    }
    return (end + 3u) / 4u * 4u;
}

// Every sample is read as part of a 32-bit word, including the last sample of
// an unaligned row. Validate the word's full extent and the shader's uint
// sample index before it can address a caller-owned buffer.
ExternalPlanes DescribeExternalPlanes(const VulkanBufferView& image) {
    const auto packed = PlaneLayout(image.format, image.size);
    const bool custom = std::any_of(
        image.planes.begin(), image.planes.end(),
        [](const VulkanPlaneLayout& p) {
            return p.offset_bytes != 0 || p.stride_bytes != 0;
        }
    );
    ExternalPlanes result;
    constexpr VkDeviceSize max = std::numeric_limits<VkDeviceSize>::max();
    for (std::size_t i = 0; i < packed.size(); ++i) {
        const PlaneGeometry& geometry = packed[i];
        const VulkanPlaneLayout& plane = image.planes[i];
        if (!geometry.present()) {
            if (custom && (plane.offset_bytes != 0 || plane.stride_bytes != 0)) {
                throw std::invalid_argument(
                    "vulkan: absent plane layout must be zero"
                );
            }
            continue;
        }
        const VkDeviceSize sample_bytes = geometry.sample_bytes;
        const VkDeviceSize row_bytes = geometry.width * sample_bytes;
        const VkDeviceSize offset =
            custom ? plane.offset_bytes : geometry.offset;
        const VkDeviceSize stride = custom ? plane.stride_bytes : row_bytes;
        if ((offset & 3u) != 0 || stride < row_bytes ||
            stride % sample_bytes != 0) {
            throw std::invalid_argument(
                "vulkan: plane offset or row stride is invalid"
            );
        }
        const VkDeviceSize stride_samples = stride / sample_bytes;
        const VkDeviceSize last_column = geometry.width - 1;
        const VkDeviceSize last_row = geometry.height - 1;
        if (last_column > UINT32_MAX || stride_samples > UINT32_MAX ||
            (last_row != 0 &&
             stride_samples > (UINT32_MAX - last_column) / last_row)) {
            throw std::invalid_argument(
                "vulkan: plane sample index exceeds shader range"
            );
        }
        const VkDeviceSize last_index = last_row * stride_samples + last_column;
        const VkDeviceSize words = last_index / (4u / sample_bytes) + 1u;
        if (words > (max - offset) / 4u) {
            throw std::invalid_argument("vulkan: plane range overflows");
        }
        const VkDeviceSize end = offset + words * 4u;
        result.offsets[i] = offset;
        result.row_samples[i] = static_cast<std::uint32_t>(stride_samples);
        result.required = std::max(result.required, end);
    }
    return result;
}

}  // namespace

void VulkanRuntime::UploadStreaming(
    const Image& original, const Image& distorted
) {
    UploadStreaming(original, distorted, {0x0f, 0x0f});
}

void VulkanRuntime::UploadStreaming(
    const Image& original, const Image& distorted,
    std::array<std::uint8_t, 2> plane_masks
) {
    // The mapped source regions can still be read by the pending submission.
    // Reject before touching either region or its associated format.
    RequireIdle();
    CheckImage(original, config_.base_size, "original");
    CheckImage(distorted, config_.base_size, "distorted");
    WriteStreamingSource(0, original, plane_masks[0]);
    WriteStreamingSource(1, distorted, plane_masks[1]);
    stream_plane_masks_ = plane_masks;
}

void VulkanRuntime::UploadStreaming(
    const ImageView& original, const ImageView& distorted
) {
    UploadStreaming(original, distorted, {0x0f, 0x0f});
}

void VulkanRuntime::UploadStreaming(
    const ImageView& original, const ImageView& distorted,
    std::array<std::uint8_t, 2> plane_masks
) {
    RequireIdle();
    if (original.size != config_.base_size ||
        distorted.size != config_.base_size) {
        throw std::invalid_argument(
            "vulkan: image view geometry does not match the pipeline"
        );
    }
    // Validate both sides before writing either mapped source region.
    const ImageViewLayout a = DescribeImageView(original);
    const ImageViewLayout b = DescribeImageView(distorted);
    (void)SelectSourceDecodeVariant(original.format, config_.chroma_filter);
    (void)SelectSourceDecodeVariant(distorted.format, config_.chroma_filter);
    PackImageView(StreamingDestination(0), original, a, plane_masks[0]);
    PackImageView(StreamingDestination(1), distorted, b, plane_masks[1]);
    stream_format_[0] = original.format;
    stream_format_[1] = distorted.format;
    stream_plane_masks_ = plane_masks;
}

void VulkanRuntime::UploadStreamingDistorted(const Image& distorted) {
    RequireIdle();
    CheckImage(distorted, config_.base_size, "distorted");
    WriteStreamingSource(1, distorted, stream_plane_masks_[1]);
}

std::byte* VulkanRuntime::StreamingDestination(std::size_t image) const {
    // Disabled streaming has no reserved source regions. Guard the shared
    // write boundary so no metric can overwrite metadata or staging tables.
    if (!config_.streaming) {
        throw std::invalid_argument("vulkan: streaming_inputs is false");
    }
    // The mapped arena or staging slot keeps the raw planes unchanged.
    const std::size_t slot = config_.stream_region[image];
    return arena_visible_
               ? reinterpret_cast<std::byte*>(arena_map_) + Bytes(slot)
               : reinterpret_cast<std::byte*>(staging_map_) +
                     Bytes(staging_source_offset(image));
}

void VulkanRuntime::WriteStreamingSource(
    std::size_t image, const Image& source, std::uint8_t plane_mask
) {
    const auto layout = PlaneLayout(source.format, source.size);
    std::byte* destination = StreamingDestination(image);
    for (std::size_t plane = 0; plane < layout.size(); ++plane) {
        const PlaneGeometry& geometry = layout[plane];
        if (!geometry.present() || (plane_mask & (1u << plane)) == 0) continue;
        std::memcpy(
            destination + geometry.offset, source.data.data() + geometry.offset,
            geometry.width * geometry.height * geometry.sample_bytes
        );
    }
    stream_format_[image] = source.format;
}

void VulkanRuntime::WriteSourceDesc(
    std::size_t image, const ImageFormat& format, std::uint32_t plane_base_word,
    const VulkanBufferView* external
) {
    // The planes follow Image's layout, whose offsets are 4-byte aligned, so a
    // word offset addresses each plane exactly.
    const std::array<PlaneGeometry, 4> layout =
        PlaneLayout(format, config_.base_size);
    std::array<std::uint32_t, 4> plane_words{};
    std::array<std::uint32_t, 4> row_samples{};
    const ExternalPlanes external_planes =
        external != nullptr ? DescribeExternalPlanes(*external)
                            : ExternalPlanes{};
    for (std::size_t plane = 0; plane < plane_words.size(); ++plane) {
        if (!layout[plane].present()) continue;
        const VkDeviceSize offset = external != nullptr
                                        ? external_planes.offsets[plane]
                                        : layout[plane].offset;
        plane_words[plane] =
            plane_base_word + static_cast<std::uint32_t>(offset / 4);
        row_samples[plane] =
            external != nullptr
                ? external_planes.row_samples[plane]
                : static_cast<std::uint32_t>(layout[plane].width);
    }
    SourceDesc desc =
        BuildSourceDesc(format, plane_words, config_.chroma_filter);
    std::copy(row_samples.begin(), row_samples.end(), desc.row_samples);
    if (arena_visible_) {
        std::memcpy(
            arena_map_ + config_.stream_desc[image], &desc, sizeof(desc)
        );
    } else {
        std::memcpy(
            staging_map_ + config_.staging_desc_offset +
                image * kSourceDescWords,
            &desc, sizeof(desc)
        );
    }
}

VkDescriptorBufferInfo VulkanRuntime::ExternalBinding(
    const VulkanBufferView& image, std::uint32_t* plane_base
) const {
    if (image.buffer == VK_NULL_HANDLE) {
        throw std::invalid_argument("vulkan: external image buffer is null");
    }
    if (image.size != config_.base_size) {
        throw std::invalid_argument(
            "vulkan: external image geometry does not match the pipeline"
        );
    }
    ValidateImage(image.format, image.size, "external image");
    (void)SelectSourceDecodeVariant(image.format, config_.chroma_filter);
    if ((image.offset & 3u) != 0) {
        throw std::invalid_argument(
            "vulkan: external image offset must be word-aligned"
        );
    }
    const ExternalPlanes planes = DescribeExternalPlanes(image);
    const VkDeviceSize required = planes.required;
    if (image.bytes < required ||
        image.bytes > std::numeric_limits<VkDeviceSize>::max() - image.offset) {
        throw std::invalid_argument(
            "vulkan: external image buffer range is too small or overflows"
        );
    }
    const VkDeviceSize alignment =
        std::max<VkDeviceSize>(ctx_->limits.minStorageBufferOffsetAlignment, 1);
    const VkDeviceSize binding_offset = image.offset - image.offset % alignment;
    const VkDeviceSize prefix = image.offset - binding_offset;
    if (required > std::numeric_limits<VkDeviceSize>::max() - prefix ||
        prefix + required > ctx_->limits.maxStorageBufferRange ||
        prefix / 4 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "vulkan: external image exceeds storage buffer range"
        );
    }
    *plane_base = static_cast<std::uint32_t>(prefix / 4);
    return VkDescriptorBufferInfo{image.buffer, binding_offset,
                                  prefix + required};
}

std::array<VulkanRuntime::Source, 2> VulkanRuntime::ResolveExternalInputs(
    const VulkanBufferView& original, const VulkanBufferView& distorted
) const {
    RequireIdle();
    const std::array<const VulkanBufferView*, 2> images{&original, &distorted};
    std::array<Source, 2> sources{};
    for (std::size_t index = 0; index < sources.size(); ++index) {
        std::uint32_t plane_base = 0;
        (void)ExternalBinding(*images[index], &plane_base);
        sources[index] = Source{Source::Kind::External, 0, images[index]};
    }
    return sources;
}

void VulkanRuntime::PrepareSourceBindings(const std::array<Source, 2>& sources) {
    RequireIdle();
    std::array<VkDescriptorBufferInfo, 2> infos{};
    for (std::size_t image = 0; image < sources.size(); ++image) {
        if (sources[image].kind == Source::Kind::External) {
            if (sources[image].external == nullptr) {
                throw std::invalid_argument("vulkan: external image is null");
            }
            std::uint32_t plane_base = 0;
            infos[image] =
                ExternalBinding(*sources[image].external, &plane_base);
        } else {
            infos[image] =
                VkDescriptorBufferInfo{arena_.buffer, 0, arena_bytes_};
        }
    }
    std::array<VkWriteDescriptorSet, 2> writes{};
    std::uint32_t count = 0;
    for (std::size_t image = 0; image < sources.size(); ++image) {
        const auto& current = source_bindings_[image];
        const auto& wanted = infos[image];
        // The caller may destroy an external buffer after Finish and create a
        // new one with the same handle value, so rewrite external descriptors
        // even when their bits match the previous submission.
        if (sources[image].kind != Source::Kind::External &&
            current.buffer == wanted.buffer && current.offset == wanted.offset &&
            current.range == wanted.range)
            continue;
        auto& write = writes[count++];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_sets_[image];
        write.dstBinding = 3;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &infos[image];
    }
    if (count != 0) {
        vkUpdateDescriptorSets(ctx_->device, count, writes.data(), 0, nullptr);
        source_bindings_ = infos;
    }
}

VulkanRuntime::ResolvedSource VulkanRuntime::ResolveSource(
    const Source& source, std::size_t image
) const {
    switch (source.kind) {
        case Source::Kind::External: {
            // The view was validated before recording. Binding offsets are
            // aligned down; plane offsets include this word-aligned prefix.
            const VkDeviceSize alignment = std::max<VkDeviceSize>(
                ctx_->limits.minStorageBufferOffsetAlignment, 1
            );
            const auto plane_base = static_cast<std::uint32_t>(
                (source.external->offset % alignment) / 4
            );
            return {&source.external->format, plane_base};
        }
        case Source::Kind::Resident:
            return {&resident_format_[source.index],
                    config_.resident_slots[source.index]};
        case Source::Kind::ImageBuffer:
            return {&input_allocations_[source.index].format,
                    input_allocations_[source.index].offset};
        case Source::Kind::Stream:
        default:
            return {&stream_format_[image], config_.stream_region[image]};
    }
}

std::array<const ImageFormat*, 2> VulkanRuntime::PublishSources(
    const std::array<Source, 2>& sources
) {
    std::array<const ImageFormat*, 2> formats{};
    for (std::size_t image = 0; image < sources.size(); ++image) {
        const ResolvedSource resolved = ResolveSource(sources[image], image);
        formats[image] = resolved.format;
        WriteSourceDesc(
            image, *resolved.format, resolved.plane_base,
            sources[image].kind == Source::Kind::External
                ? sources[image].external
                : nullptr
        );
    }
    if (arena_visible_) {
        HostToComputeBarrier();
    } else {
        const SourceCopies copies =
            PlanSourceCopies(config_, sources, formats, stream_plane_masks_);
        RecordStagedCopies(
            std::span<const CopyRange>(copies.ranges.data(), copies.count)
        );
    }
    if (std::any_of(sources.begin(), sources.end(), [](const Source& source) {
            return source.kind == Source::Kind::External;
        })) {
        // External input producers must be ordered on this queue before this
        // command buffer; the barrier makes their writes visible to prepare.
        Barrier(
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT
        );
    }
    return formats;
}

VulkanRuntime::SourceCopies VulkanRuntime::PlanSourceCopies(
    const Config& config, const std::array<Source, 2>& sources,
    const std::array<const ImageFormat*, 2>& formats,
    std::array<std::uint8_t, 2> plane_masks
) {
    SourceCopies copies;
    for (std::size_t image = 0; image < sources.size(); ++image) {
        if (sources[image].kind != Source::Kind::Stream) continue;
        copies.ranges[copies.count++] =
            CopyRange{image * kSourceFloatsPerPixel * config.base_pixels,
                      config.stream_region[image],
                      SelectedPlaneBytes(
                          *formats[image], config.base_size, plane_masks[image]
                      ) / sizeof(float)};
    }
    for (std::size_t image = 0; image < sources.size(); ++image) {
        copies.ranges[copies.count++] =
            CopyRange{config.staging_desc_offset + image * kSourceDescWords,
                      config.stream_desc[image], kSourceDescWords};
    }
    return copies;
}

std::optional<std::uint32_t> VulkanRuntime::UploadResident(
    const Image& image, std::size_t capacity
) {
    if (pending_) {
        throw std::logic_error(
            "vulkan: Upload while a submission is in flight; Finish it first"
        );
    }
    if (capacity > config_.resident_slots.size()) return std::nullopt;
    const auto free_slot =
        std::find(resident_free_.begin(), resident_free_.end(), true);
    if (free_slot == resident_free_.end()) return std::nullopt;
    const std::size_t index =
        static_cast<std::size_t>(free_slot - resident_free_.begin());

    const std::uint32_t slot = config_.resident_slots[index];
    const std::size_t bytes = image.data.size();
    if (arena_visible_) {
        std::memcpy(
            reinterpret_cast<std::uint8_t*>(arena_map_) + Bytes(slot),
            image.data.data(), bytes
        );
        resident_format_[index] = image.format;
        *free_slot = false;
        return static_cast<std::uint32_t>(index);
    }

    // Fallback: stage the raw planes and copy them in, one submit and a fence
    // wait; the memory dependency for the later scoring submission is
    // recorded with the pair. The staging buffer holds one image at word 0 and
    // the descriptors, kernels and tables after it, so an image that does not
    // fit the reserved area has to be refused rather than written over them.
    if (bytes > config_.staging_image_bytes) {
        throw std::runtime_error(
            "vulkan: this device's arena is not host visible, so a resident "
            "upload stages the image through " +
            std::to_string(config_.staging_image_bytes) + " bytes of staging (" +
            std::to_string(bytes) +
            " needed); the image does not match the geometry the pipeline was "
            "built for"
        );
    }
    std::memcpy(staging_map_, image.data.data(), bytes);
    BeginCommandBuffer();
    // RequiredBytes is a multiple of 4, so the whole buffer is whole words.
    const std::array<CopyRange, 1> copies{CopyRange{0, slot, bytes / 4}};
    RecordStagedCopies(copies);
    EndCommandBuffer();
    SubmitAndWait();
    // Publish the reservation only after the fallible GPU work succeeds, so
    // a failed upload cannot strand a backend slot without a public handle.
    resident_format_[index] = image.format;
    *free_slot = false;
    return static_cast<std::uint32_t>(index);
}

void VulkanRuntime::ReleaseResident(std::uint32_t slot) {
    RequireIdle();
    if (slot < resident_free_.size()) resident_free_[slot] = true;
}

ImageBuffer VulkanRuntime::AllocateImage(const ImageFormat& format) {
    if (pending_) {
        throw std::logic_error(
            "vulkan: AllocateImage while a submission is in flight; Finish it "
            "first"
        );
    }
    ValidateImage(format, config_.base_size, "image buffer");
    if (input_pool_size_ == 0) {
        throw std::invalid_argument(
            "vulkan: the input pool size is 0, so AllocateImage has no "
            "memory; set it to the bytes your frames need and construct again"
        );
    }
    if (!arena_visible_) {
        throw std::runtime_error(
            "vulkan: caller-owned image buffers need device memory the host "
            "can write (resizable BAR); use Upload() on this device"
        );
    }
    // Throws std::invalid_argument for a format the decode shaders cannot
    // handle, before anything is reserved.
    (void)SelectSourceDecodeVariant(format, config_.chroma_filter);
    const std::size_t needed = RequiredBytes(format, config_.base_size);
    // RequiredBytes includes whole words. Align the start, not the occupied
    // length: a single allocation must fit an exactly-sized input pool.
    const std::size_t words = needed / sizeof(float);
    if (words > input_pool_size_) {
        throw std::invalid_argument("vulkan: the input pool is full");
    }
    const std::uint64_t floats = words;

    // First fit over the live spans, which are few.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> used;
    used.reserve(input_allocations_.size());
    for (const InputAllocation& alloc : input_allocations_) {
        if (!alloc.live) continue;
        used.emplace_back(alloc.offset - input_pool_, alloc.floats);
    }
    std::sort(used.begin(), used.end());
    const auto span = FindInputSpan(used, floats, input_pool_size_);
    if (!span) {
        throw std::invalid_argument(
            "vulkan: the input pool is full; Release a buffer or raise the "
            "configured input bytes"
        );
    }

    std::size_t index = input_allocations_.size();
    bool reused = false;
    for (std::size_t i = 0; i < input_allocations_.size(); ++i) {
        if (!input_allocations_[i].live) {
            index = i;
            reused = true;
            break;
        }
    }
    if (!reused && index >= kInputIndexMask) {
        throw std::invalid_argument(
            "vulkan: maximum number of live image buffers reached"
        );
    }
    const std::uint64_t generation = NextHandleIncarnation();
    if (!reused) input_allocations_.emplace_back();
    InputAllocation& alloc = input_allocations_[index];
    alloc = InputAllocation{};
    alloc.generation = generation;
    alloc.offset = input_pool_ + static_cast<std::uint32_t>(*span);
    alloc.floats = static_cast<std::uint32_t>(floats);
    alloc.bytes = needed;
    alloc.format = format;
    alloc.live = true;
    return ImageBuffer{(static_cast<std::uint64_t>(index) + 1) |
                           (alloc.generation << kInputIndexBits),
                       config_.base_size, needed};
}

std::byte* VulkanRuntime::MapImage(ImageBuffer buffer) {
    if (pending_) {
        throw std::logic_error(
            "vulkan: MapImage while a submission is in flight; Finish it first"
        );
    }
    const std::optional<std::size_t> index = ResolveInput(buffer);
    if (!index) {
        throw std::invalid_argument(
            "vulkan: image buffer is not live (released or stale handle)"
        );
    }
    return reinterpret_cast<std::byte*>(
        arena_map_ + input_allocations_[*index].offset
    );
}

void VulkanRuntime::ReleaseInput(ImageBuffer buffer) {
    if (pending_) {
        throw std::logic_error(
            "vulkan: Release while a submission is in flight would hand the "
            "span to the next AllocateImage while it is still being read; "
            "Finish first"
        );
    }
    const std::optional<std::size_t> index = ResolveInput(buffer);
    if (!index) return;  // already released
    input_allocations_[*index].live = false;
}

std::optional<std::size_t> VulkanRuntime::ResolveInput(
    ImageBuffer buffer
) const {
    const std::uint64_t slot = buffer.slot & kInputIndexMask;
    if (slot == 0 || slot > input_allocations_.size()) return std::nullopt;
    const std::size_t candidate = static_cast<std::size_t>(slot - 1);
    const InputAllocation& alloc = input_allocations_[candidate];
    if (!alloc.live || (buffer.slot >> kInputIndexBits) != alloc.generation) {
        return std::nullopt;
    }
    return candidate;
}

std::array<VulkanRuntime::Source, 2> VulkanRuntime::ResolveInputs(
    const ImageBuffer& original, const ImageBuffer& distorted
) const {
    RequireIdle();
    const std::array<const ImageBuffer*, 2> handles{&original, &distorted};
    std::array<Source, 2> sources{};
    for (std::size_t image = 0; image < handles.size(); ++image) {
        const std::optional<std::size_t> index = ResolveInput(*handles[image]);
        if (!index) {
            throw std::invalid_argument(
                "vulkan: image buffer is not live (released or stale handle)"
            );
        }
        if (handles[image]->size != config_.base_size) {
            throw std::invalid_argument(
                "vulkan: image buffer geometry does not match the pipeline"
            );
        }
        sources[image] = Source{Source::Kind::ImageBuffer,
                                static_cast<std::uint32_t>(*index)};
    }
    return sources;
}

}  // namespace vqm::internal
