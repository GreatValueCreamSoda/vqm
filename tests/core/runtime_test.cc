// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Shared runtime contracts: handle identity, timestamp counters, optional
// device features, staging capacity, and source-copy planning.

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>

#include "handles.h"
#include "support.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "vulkan/vulkan_support.h"

VQM_TEST(TestResidentHandleIncarnations) {
    using namespace vqm;
    using namespace vqm::internal;
    ResidentTable first;
    ResidentTable second;
    first.Resize(1);
    second.Resize(1);
    constexpr Size size{8, 8};
    const UploadedImage old = first.Acquire(0, 0, size, 1);
    const UploadedImage foreign = second.Acquire(0, 0, size, 2);
    CHECK(first.Index(old).has_value());
    CHECK(!second.Index(old).has_value());
    CHECK(!first.Index(foreign).has_value());
    first.at(0).live = false;
    // Reuse past the old 20-bit limit must neither revive a stale handle nor
    // create a handle indistinguishable from the first use of the slot.
    const UploadedImage replacement = first.Acquire(0, 0, size, (1u << 20) + 1u);
    CHECK(old.slot != replacement.slot);
    CHECK(!first.Index(old).has_value());
    CHECK(first.Index(replacement).has_value());
    first.at(0).live = false;
    const UploadedImage last = first.Acquire(0, 0, size, kHandleIncarnationMax);
    CHECK(first.Index(last).has_value());
    CHECK(!first.Index(replacement).has_value());
}

VQM_TEST(TestResidentUploadPublication) {
    using namespace vqm;
    using namespace vqm::internal;
    ResidentTable resident;
    resident.Resize(1);
    constexpr Size size{8, 8};
    unsigned uploads = 0;
    const auto unavailable =
        [&](std::size_t capacity) -> std::optional<std::uint32_t> {
        ++uploads;
        CHECK(capacity == 1);
        return std::nullopt;
    };
    CHECK_THROWS_AS(resident.Upload(size, unavailable), std::invalid_argument);
    CHECK(uploads == 1);
    CHECK(!resident.at(0).live);
    CHECK(resident.RequireFree() == 0);

    const auto failed = [&](std::size_t) -> std::optional<std::uint32_t> {
        ++uploads;
        throw std::runtime_error("backend upload failed");
    };
    CHECK_THROWS_AS(resident.Upload(size, failed), std::runtime_error);
    CHECK(uploads == 2);
    CHECK(!resident.at(0).live);
    CHECK(resident.RequireFree() == 0);

    const auto uploaded =
        [&](std::size_t capacity) -> std::optional<std::uint32_t> {
        ++uploads;
        CHECK(capacity == 1);
        return 7;
    };
    const UploadedImage image = resident.Upload(size, uploaded);
    CHECK(image.size == size);
    CHECK(resident.Require(image, size, "image").device == 7);
    CHECK_THROWS_AS(resident.Upload(size, uploaded), std::invalid_argument);
    CHECK(uploads == 3);  // A full table must not invoke the backend again.
}

VQM_TEST(TestResidentReleasePublication) {
    using namespace vqm;
    using namespace vqm::internal;
    ResidentTable resident;
    ResidentTable other;
    resident.Resize(1);
    other.Resize(1);
    constexpr Size size{8, 8};
    const auto uploaded = [](std::size_t) -> std::optional<std::uint32_t> {
        return 7;
    };
    const UploadedImage image = resident.Upload(size, uploaded);
    const UploadedImage foreign = other.Upload(size, uploaded);
    unsigned releases = 0;
    const auto released = [&](std::uint32_t device) {
        ++releases;
        CHECK(device == 7);
    };
    resident.Release({}, released);
    resident.Release(foreign, released);
    CHECK(releases == 0);

    const auto failed = [&](std::uint32_t device) {
        ++releases;
        CHECK(device == 7);
        throw std::runtime_error("backend release failed");
    };
    CHECK_THROWS_AS(resident.Release(image, failed), std::runtime_error);
    CHECK(releases == 1);
    CHECK(resident.Require(image, size, "image").device == 7);

    resident.Release(image, released);
    CHECK(releases == 2);
    CHECK(!resident.Index(image).has_value());
    CHECK(resident.at(0).device == 0);
    resident.Release(image, released);
    CHECK(releases == 2);

    const UploadedImage replacement = resident.Upload(size, uploaded);
    CHECK(replacement.slot != image.slot);
    resident.Release(image, released);
    CHECK(releases == 2);
    CHECK(resident.Require(replacement, size, "image").device == 7);
    resident.Release(replacement, released);
    CHECK(releases == 3);
}

VQM_TEST(TestPipelineBatchCreationRollback) {
    using namespace vqm::internal;
    if (!vqm::test::GpuAvailable()) return;
    VulkanRuntime::Config config;
    config.base_size = {8, 8};
    config.push_constant_size = 128;
    VulkanRuntime runtime(AcquireVulkanContext(), config);
    runtime.CreateArena(1024);
    runtime.CreateDescriptors(0, 256);
    // A missing second shader fails after the first real pipeline exists.
    // Validation at device teardown also checks that rollback released it.
    CHECK_THROWS_AS(
        runtime.CreatePipelines(std::array{"reduce", "missing-test-shader"}),
        std::runtime_error
    );
    const auto pipelines = runtime.CreatePipelines(std::array{"reduce"});
    CHECK(pipelines[0] != VK_NULL_HANDLE);
    runtime.DestroyPipelines(pipelines);
}

VQM_TEST(TestTimestampCounterWrap) {
    using vqm::internal::ElapsedTimestampTicks;
    constexpr std::uint64_t wrap36 = std::uint64_t{1} << 36;
    CHECK(ElapsedTimestampTicks(10, 25, 36) == 15);
    CHECK(ElapsedTimestampTicks(wrap36 - 5, 7, 36) == 12);
    CHECK(ElapsedTimestampTicks(wrap36 + 10, 25, 36) == 15);
    CHECK(ElapsedTimestampTicks(UINT64_MAX - 4, 7, 64) == 12);
}

VQM_TEST(TestBorrowedOptionalFeatureFlags) {
    using namespace vqm;
    using namespace vqm::internal;
    if (!test::GpuAvailable()) return;
    const auto owner = AcquireVulkanContext();
    CHECK_THROWS_AS(ShutdownDefaultVulkanContext(), std::logic_error);
    VulkanDevice device{owner->physical_device, owner->device, owner->queue,
                        owner->queue_family, owner->sixteen_bit_storage};
    const auto conservative = BorrowVulkanContext(device);
    CHECK(!conservative->shader_float16);
    CHECK(conservative->timestamp_valid_bits == owner->timestamp_valid_bits);
    if (owner->shader_float16) {
        device.shader_float16 = true;
        CHECK(BorrowVulkanContext(device)->shader_float16);
    }
}

VQM_TEST(TestStagingSourceSizing) {
    using vqm::internal::PlanStagingSources;
    constexpr std::size_t pixels = 64;
    constexpr std::size_t image_floats = 4 * pixels;
    const auto none = PlanStagingSources(pixels, false, 0);
    CHECK(none.reserved_floats == 0);
    CHECK(none.image_bytes == 0);
    const auto resident = PlanStagingSources(pixels, false, 2);
    CHECK(resident.reserved_floats == image_floats);
    CHECK(resident.image_bytes == image_floats * sizeof(float));
    for (const std::size_t slots : {0u, 2u}) {
        const auto streaming = PlanStagingSources(pixels, true, slots);
        CHECK(streaming.reserved_floats == 2 * image_floats);
        CHECK(streaming.image_bytes == image_floats * sizeof(float));
    }
}

// The non-host-visible arena path stages only stream pixels, followed by both
// source descriptors. This test does not require a device with that memory
// topology: it exercises the copy planner directly.
VQM_TEST(TestSourceCopyPlanning) {
    using Runtime = vqm::internal::VulkanRuntime;
    using Source = Runtime::Source;
    using Kind = Source::Kind;

    Runtime::Config config;
    config.base_size = {8, 8};
    config.base_pixels = 64;
    config.stream_region = {101, 201};
    config.stream_desc = {501, 601};
    config.staging_desc_offset = 800;
    const auto byte_format =
        vqm::test::PinnedFormat(vqm::AlphaType::None, vqm::PixelType::Byte);
    const auto word_format =
        vqm::test::PinnedFormat(vqm::AlphaType::None, vqm::PixelType::Word);
    const std::array<const vqm::ImageFormat*, 2> formats{&byte_format,
                                                         &word_format};
    const auto check = [&](const std::array<Source, 2>& sources,
                           std::initializer_list<Runtime::CopyRange> expected) {
        const auto copies = Runtime::PlanSourceCopies(config, sources, formats);
        CHECK(copies.count == expected.size());
        if (copies.count != expected.size()) return;
        std::size_t i = 0;
        for (const auto& copy : expected) {
            CHECK(copies.ranges[i].staging_floats == copy.staging_floats);
            CHECK(copies.ranges[i].arena_floats == copy.arena_floats);
            CHECK(copies.ranges[i].floats == copy.floats);
            ++i;
        }
    };

    // 8x8 RGB occupies 48 words at 8 bits and 96 words at 16 bits.
    constexpr std::size_t desc_words = vqm::internal::kSourceDescWords;
    check(
        {Source{Kind::Stream, 0}, Source{Kind::Stream, 0}},
        {{0, 101, 48},
         {256, 201, 96},
         {800, 501, desc_words},
         {800 + desc_words, 601, desc_words}}
    );
    check(
        {Source{Kind::Stream, 0}, Source{Kind::Resident, 2}},
        {{0, 101, 48},
         {800, 501, desc_words},
         {800 + desc_words, 601, desc_words}}
    );
    check(
        {Source{Kind::Resident, 2}, Source{Kind::Stream, 0}},
        {{256, 201, 96},
         {800, 501, desc_words},
         {800 + desc_words, 601, desc_words}}
    );
    check(
        {Source{Kind::Resident, 2}, Source{Kind::ImageBuffer, 3}},
        {{800, 501, desc_words}, {800 + desc_words, 601, desc_words}}
    );

    // Luma-only copies keep the first plane's 64 bytes at 8 bits and
    // 128 bytes at 16 bits, without staging either chroma plane.
    const auto luma = Runtime::PlanSourceCopies(
        config, {Source{Kind::Stream, 0}, Source{Kind::Stream, 0}}, formats,
        {0x01, 0x01}
    );
    CHECK(luma.count == 4);
    if (luma.count == 4) {
        CHECK(luma.ranges[0].staging_floats == 0);
        CHECK(luma.ranges[0].arena_floats == 101);
        CHECK(luma.ranges[0].floats == 16);
        CHECK(luma.ranges[1].staging_floats == 256);
        CHECK(luma.ranges[1].arena_floats == 201);
        CHECK(luma.ranges[1].floats == 32);
        CHECK(luma.ranges[2].staging_floats == 800);
        CHECK(luma.ranges[2].arena_floats == 501);
        CHECK(luma.ranges[3].staging_floats == 800 + desc_words);
        CHECK(luma.ranges[3].arena_floats == 601);
    }
}
