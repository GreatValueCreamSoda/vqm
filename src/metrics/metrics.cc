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
#include "metrics/psnr_hvs/layout.h"
#include "metrics/vmaf/features/vmaf_preprocess.h"
#include "vqm_shared.h"
#include "vqm_views.h"
#include "vqm_vulkan.h"
#include "vulkan/vulkan_runtime.h"

namespace vqm {
namespace {

struct SelectedConfigs {
    std::optional<Ssimulacra2Config> ssimulacra2;
    std::optional<ButteraugliConfig> butteraugli;
    std::optional<VmafConfig> vmaf_v0;
    std::optional<VmafV1Config> vmaf_v1;
    std::optional<CvvdpConfig> cvvdp;
    std::optional<PsnrHvsConfig> psnr_hvs;
    std::optional<SharedMetricConfig> spatial_shared;
};

SelectedConfigs SelectConfigs(const MetricsConfig& config) {
    if (!config.ssimulacra2 && !config.butteraugli && !config.vmaf_v0 &&
        !config.vmaf_v1 && !config.cvvdp && !config.psnr_hvs)
        throw std::invalid_argument("metrics: select at least one metric");
    if (!std::isfinite(config.background) || config.background < 0.0f ||
        config.background > 1.0f)
        throw std::invalid_argument(
            "metrics: background must be between 0 and 1"
        );
    internal::ValidateImage(
        config.original_format, config.size, "original format"
    );
    internal::ValidateImage(
        config.distorted_format, config.size, "distorted format"
    );
    if (config.vmaf_v0 || config.vmaf_v1) {
        internal::ValidateVmafSdrFormat(config.original_format);
        internal::ValidateVmafSdrFormat(config.distorted_format);
    }
    if (config.cvvdp) {
        internal::ValidateCvvdpInput(
            config.original_format, config.cvvdp->display
        );
        internal::ValidateCvvdpInput(
            config.distorted_format, config.cvvdp->display
        );
    }
    if (config.psnr_hvs) {
        internal::ValidatePsnrHvsFormat(config.original_format, config.size);
        internal::ValidatePsnrHvsFormat(config.distorted_format, config.size);
        if (!internal::SameImageFormat(
                config.original_format, config.distorted_format
            ))
            throw std::invalid_argument(
                "metrics: PSNR-HVS requires matching native input formats"
            );
    }

    SelectedConfigs result;
    result.ssimulacra2 = config.ssimulacra2;
    result.butteraugli = config.butteraugli;
    result.vmaf_v0 = config.vmaf_v0;
    result.vmaf_v1 = config.vmaf_v1;
    result.cvvdp = config.cvvdp;
    result.psnr_hvs = config.psnr_hvs;
    // Unified evaluation accepts streamed or borrowed inputs. Resident pools
    // belong to the individual metric APIs.
    const auto configure_inputs = [&](auto& metric) {
        metric.size = config.size;
        metric.streaming_inputs = config.streaming_inputs;
        metric.memory_policy = config.memory_policy;
        metric.gpu_resident_images = 0;
    };
    if (result.ssimulacra2) {
        auto& metric = *result.ssimulacra2;
        configure_inputs(metric);
        metric.input_bytes = 0;
    }
    if (result.butteraugli) {
        auto& metric = *result.butteraugli;
        configure_inputs(metric);
        metric.input_bytes = 0;
    }
    if (result.vmaf_v0) {
        auto& metric = *result.vmaf_v0;
        configure_inputs(metric);
        metric.background = config.background;
    }
    if (result.vmaf_v1) {
        auto& metric = *result.vmaf_v1;
        configure_inputs(metric);
        metric.input_format = config.original_format;
        metric.background = config.background;
        if (metric.chroma_sampling == VmafChromaSampling::Native &&
            !internal::SameImageFormat(
                config.original_format, config.distorted_format
            ))
            throw std::invalid_argument(
                "metrics: VMAF v1 Native requires matching input formats"
            );
    }
    if (result.cvvdp) configure_inputs(*result.cvvdp);
    if (result.psnr_hvs) {
        configure_inputs(*result.psnr_hvs);
        result.psnr_hvs->input_bytes = 0;
    }
    if (result.ssimulacra2 && result.butteraugli) {
        SharedMetricConfig c;
        c.size = config.size;
        c.streaming_inputs = config.streaming_inputs;
        c.memory_policy = config.memory_policy;
        c.half_planes = result.ssimulacra2->half_planes;
        c.chroma_filter = result.ssimulacra2->chroma_filter;
        c.butter_chroma_filter = result.butteraugli->chroma_filter;
        c.hf_asymmetry = result.butteraugli->hf_asymmetry;
        c.xmul = result.butteraugli->xmul;
        c.intensity_target = result.butteraugli->intensity_target;
        result.spatial_shared = c;
    }
    return result;
}

template <class Config>
MetricMemoryRequirements Requirements(
    const Config& config, const VulkanDevice* device
) {
    return device ? GetMemoryRequirements(config, *device)
                  : GetMemoryRequirements(config);
}

MetricsMemoryRequirements PlanMemory(
    const MetricsConfig& config, const VulkanDevice* device
) {
    const auto selected = SelectConfigs(config);
    MetricsMemoryRequirements result;
    if (selected.spatial_shared)
        result.spatial = Requirements(*selected.spatial_shared, device);
    else if (selected.ssimulacra2)
        result.spatial = Requirements(*selected.ssimulacra2, device);
    else if (selected.butteraugli)
        result.spatial = Requirements(*selected.butteraugli, device);
    if (selected.vmaf_v0)
        result.vmaf_v0 = Requirements(*selected.vmaf_v0, device);
    if (selected.vmaf_v1)
        result.vmaf_v1 = Requirements(*selected.vmaf_v1, device);
    if (selected.cvvdp) result.cvvdp = Requirements(*selected.cvvdp, device);
    if (selected.psnr_hvs)
        result.psnr_hvs = Requirements(*selected.psnr_hvs, device);
    return result;
}

template <class Pipeline, class Config>
std::unique_ptr<Pipeline> MakePipeline(
    const Config& config, const VulkanDevice* device
) {
    return device ? std::make_unique<Pipeline>(config, *device)
                  : std::make_unique<Pipeline>(config);
}

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

MetricsMemoryRequirements GetMemoryRequirements(const MetricsConfig& config) {
    return PlanMemory(config, nullptr);
}
MetricsMemoryRequirements GetMemoryRequirements(
    const MetricsConfig& config, const VulkanDevice& device
) {
    return PlanMemory(config, &device);
}
std::size_t MaxWorkersForBudgetWorstCase(
    const MetricsMemoryRequirements& requirements,
    std::span<const std::uint64_t> heap_budgets
) {
    std::vector<MetricMemoryRequirements> components;
    components.reserve(5);
    if (requirements.spatial) components.push_back(*requirements.spatial);
    if (requirements.vmaf_v0) components.push_back(*requirements.vmaf_v0);
    if (requirements.vmaf_v1) components.push_back(*requirements.vmaf_v1);
    if (requirements.cvvdp) components.push_back(*requirements.cvvdp);
    if (requirements.psnr_hvs) components.push_back(*requirements.psnr_hvs);
    return MaxWorkersForBudgetWorstCase(
        std::span<const MetricMemoryRequirements>(components), heap_budgets
    );
}

struct Metrics::Impl {
    Impl(MetricsConfig c, const VulkanDevice* device) : config(std::move(c)) {
        const auto selected = SelectConfigs(config);
        if (selected.spatial_shared)
            spatial = MakePipeline<SharedMetricWorkspace>(
                *selected.spatial_shared, device
            );
        else if (selected.ssimulacra2)
            ssimu =
                MakePipeline<Ssimulacra2Pipeline>(*selected.ssimulacra2, device);
        else if (selected.butteraugli)
            butter =
                MakePipeline<ButteraugliPipeline>(*selected.butteraugli, device);
        if (selected.vmaf_v0)
            v0 = MakePipeline<VmafPipeline>(*selected.vmaf_v0, device);
        if (selected.vmaf_v1)
            v1 = MakePipeline<VmafV1Pipeline>(*selected.vmaf_v1, device);
        if (selected.cvvdp)
            cvvdp = MakePipeline<CvvdpPipeline>(*selected.cvvdp, device);
        if (selected.psnr_hvs)
            psnr_hvs = MakePipeline<PsnrHvsPipeline>(*selected.psnr_hvs, device);
    }

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
            if (spatial)
                return spatial
                    ->ComputeSsimulacra2(
                        original, distorted, submission, config.background
                    )
                    .Score();
            else
                return ssimu
                    ->Finish(ssimu->Submit(
                        original, distorted, submission, config.background
                    ))
                    .Score();
        } else {
            if (spatial)
                return spatial
                    ->ComputeSsimulacra2(original, distorted, config.background)
                    .Score();
            else
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
            if (spatial)
                return spatial
                    ->ComputeButteraugli(
                        original, distorted, submission, config.background
                    )
                    .score;
            else
                return butter
                    ->Finish(butter->Submit(
                        original, distorted, submission, config.background
                    ))
                    .score;
        } else {
            if (spatial)
                return spatial
                    ->ComputeButteraugli(original, distorted, config.background)
                    .score;
            else
                return butter->Compute(original, distorted, config.background)
                    .score;
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
        if (finished) return std::nullopt;
        try {
            if (v0) AddFrame(v0->Flush(), false);
            if (v1) AddFrame(v1->Flush(), true);
            if (cvvdp) final_cvvdp = cvvdp->Finalize();
            finished = true;
            auto result = TakeReady();
            if (!pending.empty())
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

    MetricsConfig config;
    std::unique_ptr<SharedMetricWorkspace> spatial;
    std::unique_ptr<Ssimulacra2Pipeline> ssimu;
    std::unique_ptr<ButteraugliPipeline> butter;
    std::unique_ptr<VmafPipeline> v0;
    std::unique_ptr<VmafV1Pipeline> v1;
    std::unique_ptr<CvvdpPipeline> cvvdp;
    std::unique_ptr<PsnrHvsPipeline> psnr_hvs;
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
    return {std::move(tail), impl_->final_cvvdp, impl_->next_index};
}
void Metrics::Reset() { impl_->Reset(); }
MetricsMemoryUsage Metrics::GetMemoryUsage() const {
    MetricsMemoryUsage result;
    if (impl_->spatial)
        result.spatial = impl_->spatial->GetMemoryUsage();
    else if (impl_->ssimu)
        result.spatial = impl_->ssimu->GetMemoryUsage();
    else if (impl_->butter)
        result.spatial = impl_->butter->GetMemoryUsage();
    if (impl_->v0) result.vmaf_v0 = impl_->v0->GetMemoryUsage();
    if (impl_->v1) result.vmaf_v1 = impl_->v1->GetMemoryUsage();
    if (impl_->cvvdp) result.cvvdp = impl_->cvvdp->GetMemoryUsage();
    if (impl_->psnr_hvs) result.psnr_hvs = impl_->psnr_hvs->GetMemoryUsage();
    return result;
}
}  // namespace vqm
