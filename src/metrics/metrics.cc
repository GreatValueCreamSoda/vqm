// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include "vqm_metrics.h"

#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "image_view.h"
#include "input.h"
#include "metrics/cvvdp/cvvdp_internal.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"
#include "metrics/metric_pipelines_internal.h"

namespace vqm {
namespace {

void ValidatePairFormat(
    Size size, const ImageFormat& original_format,
    const ImageFormat& distorted_format, Size original_size,
    const ImageFormat& actual_original, Size distorted_size,
    const ImageFormat& actual_distorted
) {
    if (size.width != original_size.width ||
        size.height != original_size.height ||
        size.width != distorted_size.width ||
        size.height != distorted_size.height)
        throw std::invalid_argument(
            "metrics: image size differs from configured size"
        );
    if (!internal::SameImageFormat(original_format, actual_original) ||
        !internal::SameImageFormat(distorted_format, actual_distorted))
        throw std::invalid_argument(
            "metrics: image format differs from configured format"
        );
}

void ValidateStorage(const Image& image, Size size, const char* name) {
    internal::CheckImage(image, size, name);
}
void ValidateStorage(const ImageView& image, Size, const char*) {
    (void)internal::DescribeImageView(image);
}
void ValidateStorage(const VulkanBufferView& image, Size, const char* name) {
    internal::ValidateImage(image.format, image.size, name);
    if (image.buffer == VK_NULL_HANDLE || image.offset % 4 != 0)
        throw std::invalid_argument(
            "metrics: invalid Vulkan input buffer or offset"
        );
}

ImageView PackedView(
    std::span<const std::byte> bytes, const ImageFormat& format, Size size
) {
    if (bytes.size() != RequiredBytes(format, size))
        throw std::invalid_argument("metrics: packed input has incorrect size");
    ImageView view;
    view.size = size;
    view.format = format;
    const auto planes = internal::PlaneLayout(format, size);
    for (std::size_t i = 0; i < planes.size(); ++i) {
        if (!planes[i].present()) continue;
        view.planes[i] = {bytes.data() + planes[i].offset,
                          planes[i].width * planes[i].sample_bytes};
    }
    return view;
}

}  // namespace

struct Metrics::Impl : MetricPipelines::Impl {
    Impl(MetricsConfig pipeline_config, const VulkanDevice* device)
        : MetricPipelines::Impl(std::move(pipeline_config), device, true) {}

    void RequireOpen() const {
        if (failed)
            throw std::logic_error("metrics: previous call failed; call Reset");
        if (finished)
            throw std::logic_error("metrics: sequence finished; call Reset");
    }

    bool Complete(const MetricsResult& value) const {
        return (!config.ssimulacra2 || value.ssimulacra2.has_value()) &&
               (!config.butteraugli || value.butteraugli.has_value()) &&
               (!config.vmaf_v0 || value.vmaf_v0.has_value()) &&
               (!config.vmaf_v1 || value.vmaf_v1.has_value()) &&
               (!config.psnr_hvs || value.psnr_hvs.has_value());
    }

    template <class Frame>
    void AddFrame(const std::optional<Frame>& frame, bool version_one) {
        if (!frame) return;
        for (auto& value : pending) {
            if (value.frame_index != frame->index) continue;
            auto& target = version_one ? value.vmaf_v1 : value.vmaf_v0;
            if (target) throw std::logic_error("metrics: duplicate VMAF frame");
            target = frame->score;
            return;
        }
        throw std::logic_error("metrics: unexpected VMAF frame index");
    }

    std::optional<MetricsResult> TakeReady() {
        if (pending.empty() || !Complete(pending.front())) return std::nullopt;
        auto value = std::move(pending.front());
        pending.pop_front();
        return value;
    }

    // Spatial input/ownership dispatch is separate from frame alignment and
    // semaphore sequencing, which remain visible together in Run below.
    template <class Input>
    double ComputeSsimulacra2(
        const Input& original, const Input& distorted,
        const VulkanSubmitSync& submission
    ) {
        if constexpr (std::is_same_v<Input, VulkanBufferView>) {
            return ssimu
                ->Finish(ssimu->Submit(
                    original, distorted, submission, config.background
                ))
                .Score();
        } else {
            return ssimu->Compute(original, distorted, config.background)
                .Score();
        }
    }

    template <class Input>
    double ComputeButteraugli(
        const Input& original, const Input& distorted,
        const VulkanSubmitSync& submission
    ) {
        if constexpr (std::is_same_v<Input, VulkanBufferView>) {
            return butter
                ->Finish(butter->Submit(
                    original, distorted, submission, config.background
                ))
                .score;
        } else {
            return butter->Compute(original, distorted, config.background).score;
        }
    }

    template <class Input>
    std::optional<MetricsResult> Run(
        const Input& original, const Input& distorted,
        const VulkanSubmitSync* sync = nullptr, bool image = false
    ) {
        RequireOpen();
        if (next_index == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("metrics: too many frames");
        ValidatePairFormat(
            config.size, config.original_format, config.distorted_format,
            original.size, original.format, distorted.size, distorted.format
        );
        ValidateStorage(original, config.size, "original");
        ValidateStorage(distorted, config.size, "distorted");
        if constexpr (!std::is_same_v<Input, VulkanBufferView>) {
            if (!config.streaming_inputs)
                throw std::invalid_argument(
                    "metrics: streaming inputs are disabled"
                );
        }
        if (sync) internal::VulkanRuntime::ValidateSubmitSync(*sync);

        const unsigned count =
            static_cast<unsigned>(!!config.ssimulacra2) +
            static_cast<unsigned>(!!config.butteraugli) +
            static_cast<unsigned>(!!v0) + static_cast<unsigned>(!!v1) +
            static_cast<unsigned>(!!cvvdp) + static_cast<unsigned>(!!psnr_hvs);
        unsigned index = 0;
        const auto next_sync = [&]() {
            VulkanSubmitSync result{};
            if (sync && index == 0) result.waits = sync->waits;
            if (sync && index + 1 == count) result.signals = sync->signals;
            ++index;
            return result;
        };
        try {
            MetricsResult value;
            value.frame_index = next_index;
            pending.push_back(value);
            auto& current = pending.back();
            if (config.ssimulacra2) {
                const auto submission = next_sync();
                current.ssimulacra2 =
                    ComputeSsimulacra2(original, distorted, submission);
            }
            if (config.butteraugli) {
                const auto submission = next_sync();
                current.butteraugli =
                    ComputeButteraugli(original, distorted, submission);
            }
            if (v0) {
                const auto submission = next_sync();
                if constexpr (std::is_same_v<Input, VulkanBufferView>)
                    AddFrame(v0->Push(original, distorted, submission), false);
                else
                    AddFrame(v0->Push(original, distorted), false);
            }
            if (v1) {
                const auto submission = next_sync();
                if constexpr (std::is_same_v<Input, VulkanBufferView>)
                    AddFrame(v1->Push(original, distorted, submission), true);
                else
                    AddFrame(v1->Push(original, distorted), true);
            }
            if (cvvdp) {
                const auto submission = next_sync();
                if constexpr (std::is_same_v<Input, VulkanBufferView>) {
                    if (image)
                        current.cvvdp =
                            cvvdp->ComputeImage(original, distorted, submission)
                                .score;
                    else
                        cvvdp->Push(original, distorted, submission);
                } else {
                    if (image)
                        current.cvvdp =
                            cvvdp->ComputeImage(original, distorted).score;
                    else
                        cvvdp->Push(original, distorted);
                }
            }
            if (psnr_hvs) {
                const auto submission = next_sync();
                if constexpr (std::is_same_v<Input, VulkanBufferView>)
                    current.psnr_hvs = psnr_hvs
                                           ->Finish(psnr_hvs->Submit(
                                               original, distorted, submission
                                           ))
                                           .score;
                else
                    current.psnr_hvs =
                        psnr_hvs->Compute(original, distorted).score;
            }
            ++next_index;
            return TakeReady();
        } catch (...) {
            failed = true;
            throw;
        }
    }

    template <class Input>
    MetricsResult RunImage(
        const Input& original, const Input& distorted,
        const VulkanSubmitSync* sync = nullptr
    ) {
        RequireOpen();
        if (next_index != 0)
            throw std::logic_error(
                "metrics: reset the active sequence before ComputeImage"
            );
        auto first = Run(original, distorted, sync, true);
        auto last = Flush();
        if (first && last)
            throw std::logic_error(
                "metrics: unexpected extra still-image result"
            );
        if (!first && !last)
            throw std::logic_error("metrics: missing still-image result");
        MetricsResult result = first ? std::move(*first) : std::move(*last);
        Reset();
        return result;
    }

    std::optional<MetricsResult> Flush() {
        if (failed)
            throw std::logic_error("metrics: previous call failed; call Reset");
        try {
            if (!finished) {
                if (v0) AddFrame(v0->Flush(), false);
                if (v1)
                    while (const auto frame = v1->Flush()) AddFrame(frame, true);
                if (cvvdp) final_cvvdp = cvvdp->Finalize();
                finished = true;
            }
            auto result = TakeReady();
            if (!result && !pending.empty())
                throw std::logic_error("metrics: incomplete frame at flush");
            return result;
        } catch (...) {
            failed = true;
            throw;
        }
    }

    void Reset() {
        if (v0) v0->Reset();
        if (v1) v1->Reset();
        if (cvvdp) cvvdp->Reset();
        final_cvvdp.reset();
        pending.clear();
        next_index = 0;
        failed = false;
        finished = false;
    }

    std::optional<CvvdpScore> final_cvvdp;
    std::deque<MetricsResult> pending;
    std::uint64_t next_index = 0;
    bool failed = false;
    bool finished = false;
};

Metrics::Metrics(MetricsConfig config)
    : impl_(std::make_unique<Impl>(std::move(config), nullptr)) {}
Metrics::Metrics(MetricsConfig config, const VulkanDevice& device)
    : impl_(std::make_unique<Impl>(std::move(config), &device)) {}
Metrics::~Metrics() = default;
Metrics::Metrics(Metrics&&) noexcept = default;
Metrics& Metrics::operator=(Metrics&&) noexcept = default;
std::optional<MetricsResult> Metrics::Compute(
    const Image& original, const Image& distorted
) {
    return impl_->Run(original, distorted);
}
std::optional<MetricsResult> Metrics::Compute(
    const ImageView& original, const ImageView& distorted
) {
    return impl_->Run(original, distorted);
}
std::optional<MetricsResult> Metrics::Compute(
    std::span<const std::byte> original, std::span<const std::byte> distorted
) {
    const auto a =
        PackedView(original, impl_->config.original_format, impl_->config.size);
    const auto b = PackedView(
        distorted, impl_->config.distorted_format, impl_->config.size
    );
    return impl_->Run(a, b);
}
std::optional<MetricsResult> Metrics::Compute(
    const VulkanBufferView& original, const VulkanBufferView& distorted
) {
    return impl_->Run(original, distorted);
}
std::optional<MetricsResult> Metrics::Compute(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    const VulkanSubmitSync& sync
) {
    return impl_->Run(original, distorted, &sync);
}
MetricsResult Metrics::ComputeImage(
    const Image& original, const Image& distorted
) {
    return impl_->RunImage(original, distorted);
}
MetricsResult Metrics::ComputeImage(
    const ImageView& original, const ImageView& distorted
) {
    return impl_->RunImage(original, distorted);
}
MetricsResult Metrics::ComputeImage(
    std::span<const std::byte> original, std::span<const std::byte> distorted
) {
    const auto a =
        PackedView(original, impl_->config.original_format, impl_->config.size);
    const auto b = PackedView(
        distorted, impl_->config.distorted_format, impl_->config.size
    );
    return impl_->RunImage(a, b);
}
MetricsResult Metrics::ComputeImage(
    const VulkanBufferView& original, const VulkanBufferView& distorted
) {
    return impl_->RunImage(original, distorted);
}
MetricsResult Metrics::ComputeImage(
    const VulkanBufferView& original, const VulkanBufferView& distorted,
    const VulkanSubmitSync& sync
) {
    return impl_->RunImage(original, distorted, &sync);
}
std::optional<MetricsResult> Metrics::Flush() { return impl_->Flush(); }
MetricsFinalResult Metrics::Finalize() {
    auto tail = impl_->Flush();
    auto last = impl_->Flush();
    if (!last)
        return {std::move(tail), impl_->final_cvvdp, impl_->next_index, {}};
    return {std::move(last), impl_->final_cvvdp, impl_->next_index,
            std::move(tail)};
}
void Metrics::Reset() { impl_->Reset(); }
MetricsMemoryUsage Metrics::GetMemoryUsage() const {
    return impl_->GetMemoryUsage();
}
}  // namespace vqm
