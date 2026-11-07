// Precision research probe, promoted from
// build/reports/vmaf-fp16/sequence_probe.cc. Records features only; timing
// claims must use the timed benchmark driver.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "vmaf.h"
#include "vmaf_v1.h"
#include "vulkan/vulkan_support.h"

namespace {

using Fields = std::array<std::string, 14>;
constexpr std::array<unsigned, 6> kFrameSteps{0, 1, 2, 1, 0, 3};
constexpr std::array kSceneNames{"flat-step",        "lowbit-motion",
                                 "textured-lowbits", "near-black",
                                 "chroma-texture",   "extreme-motion"};

std::string FormatNumber(double x) {
    std::ostringstream s;
    s << std::setprecision(17) << x;
    return s.str();
}

void PrintCsvRow(const Fields& fields) {
    bool first = true;
    for (const auto& field : fields) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '"';
        for (char c : field) {
            if (c == '"') std::cout << '"';
            std::cout << c;
        }
        std::cout << '"';
    }
    std::cout << '\n';
}

std::uint32_t HashNoise(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Native planar YUV444 samples retain low code bits at each supported depth.
// The synthetic arithmetic is kept identical to the original precision study.
vqm::Image GenerateFrame(
    unsigned depth, unsigned scene, unsigned frame, bool distorted
) {
    vqm::Image image;
    image.size = {320, 192};
    image.format.type = depth == 8 ? vqm::PixelType::Byte : vqm::PixelType::Word;
    image.format.depth = static_cast<std::uint8_t>(depth);
    image.format.family = vqm::ColorFamily::Yuv;
    image.format.matrix = vqm::MatrixCoefficients::Bt709;
    image.format.transfer = vqm::TransferCharacteristics::Bt709;
    image.format.range = vqm::PixelRange::Limited;
    image.data.resize(vqm::RequiredBytes(image.format, image.size));
    const unsigned shift = depth - 8, maximum = (1u << depth) - 1u;
    for (unsigned channel = 0; channel < 3; ++channel) {
        for (unsigned y = 0; y < image.size.height; ++y) {
            for (unsigned x = 0; x < image.size.width; ++x) {
                const unsigned pos = y * image.size.width + x;
                unsigned code = 128u << shift;
                if (channel == 0) {
                    switch (scene) {
                        case 0:
                            code = (228u << shift) +
                                   (shift ? 1u << (shift - 1) : 0u);
                            code += kFrameSteps[frame] - unsigned(distorted);
                            break;
                        case 1:
                            code = (unsigned(depth == 8 ? 253 : 255) << shift) +
                                   kFrameSteps[frame];
                            code -= unsigned(distorted);
                            break;
                        case 2:
                            code = ((64u + (x * 11u + y * 7u) % 128u) << shift) +
                                   kFrameSteps[frame];
                            code += unsigned(distorted);
                            break;
                        case 3:
                            code = 1u + (x + 3u * y) % 4u + kFrameSteps[frame] +
                                   unsigned(distorted);
                            break;
                        case 4:
                            code = (64u +
                                    HashNoise(x + 331u * y + 17u * frame) % 128u)
                                   << shift;
                            code += unsigned(distorted);
                            break;
                        default:
                            code = frame % 2 ? maximum : 0u;
                            break;
                    }
                } else if (scene == 4) {
                    const unsigned unit = 1u << (shift > 3u ? shift - 3u : 0u);
                    const unsigned noise =
                        HashNoise(x + 331u * y + 19u * channel + 41u * frame) %
                        64u;
                    code = (96u << shift) + noise * unit + unsigned(distorted);
                }
                code = std::min(code, maximum);
                const unsigned sample =
                    channel * image.size.width * image.size.height + pos;
                if (depth == 8)
                    image.data[sample] = static_cast<std::uint8_t>(code);
                else {
                    const auto word = static_cast<std::uint16_t>(code);
                    std::memcpy(
                        image.data.data() + 2u * sample, &word, sizeof(word)
                    );
                }
            }
        }
    }
    return image;
}

bool PrintDeviceCapabilities() {
    auto context = vqm::internal::AcquireVulkanContext();
    if (!context) return false;
    VkPhysicalDeviceFloatControlsProperties controls{};
    controls.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &controls;
    vkGetPhysicalDeviceProperties2(context->physical_device, &properties);
    std::cerr << "shaderFloat16=" << context->shader_float16
              << " storage16=" << context->sixteen_bit_storage
              << " denormPreserve16=" << controls.shaderDenormPreserveFloat16
              << " denormFlush16=" << controls.shaderDenormFlushToZeroFloat16
              << " RTE16=" << controls.shaderRoundingModeRTEFloat16
              << " RTZ16=" << controls.shaderRoundingModeRTZFloat16 << '\n';
    return true;
}

void ProbeVmaf(unsigned depth, unsigned scene, Fields& fields, bool clip) {
    const auto first = GenerateFrame(depth, scene, 0, false);
    vqm::VmafConfig config;
    config.size = first.size;
    config.gpu_resident_images = 0;
    config.clip = clip;
    vqm::VmafPipeline pipeline(config);
    auto emit = [&](const std::optional<vqm::VmafFrame>& frame) {
        if (!frame) return;
        fields[3] = std::to_string(frame->index);
        fields[4] = FormatNumber(frame->score);
        fields[5] = FormatNumber(frame->adm2);
        fields[6] = FormatNumber(frame->motion2);
        for (unsigned feature = 0; feature < 4; ++feature)
            fields[7 + feature] = FormatNumber(frame->vif[feature]);
        PrintCsvRow(fields);
    };
    for (unsigned frame = 0; frame < kFrameSteps.size(); ++frame)
        emit(pipeline.Push(
            GenerateFrame(depth, scene, frame, false),
            GenerateFrame(depth, scene, frame, true)
        ));
    emit(pipeline.Flush());
}

void ProbeVmafV1(unsigned depth, unsigned scene, Fields& fields, bool clip) {
    const auto first = GenerateFrame(depth, scene, 0, false);
    vqm::VmafV1Config config;
    config.size = first.size;
    config.input_format = first.format;
    config.gpu_resident_images = 0;
    config.clip = clip;
    vqm::VmafV1Pipeline pipeline(config);
    auto emit = [&](const std::optional<vqm::VmafV1Frame>& frame) {
        if (!frame) return;
        fields[3] = std::to_string(frame->index);
        fields[4] = FormatNumber(frame->score);
        fields[5] = FormatNumber(frame->adm3);
        fields[6] = FormatNumber(frame->motion3);
        fields[11] = FormatNumber(frame->speed_chroma_uv);
        fields[12] = FormatNumber(frame->cambi);
        PrintCsvRow(fields);
    };
    for (unsigned frame = 0; frame < kFrameSteps.size(); ++frame)
        emit(pipeline.Push(
            GenerateFrame(depth, scene, frame, false),
            GenerateFrame(depth, scene, frame, true)
        ));
    emit(pipeline.Flush());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--unclipped")) {
        std::cerr << "usage: " << argv[0] << " [--unclipped]\n";
        return 1;
    }
    const bool clip = argc == 1;
    if (!PrintDeviceCapabilities()) return 2;
    PrintCsvRow(
        {"case", "metric", "depth", "frame", "score", "adm", "motion", "vif0",
         "vif1", "vif2", "vif3", "speed", "cambi", "error"}
    );
    for (unsigned depth : {8u, 10u, 12u, 16u}) {
        for (unsigned scene = 0; scene < kSceneNames.size(); ++scene) {
            for (unsigned metric = 0; metric < 2; ++metric) {
                Fields fields{kSceneNames[scene], metric ? "v1" : "v0",
                              std::to_string(depth)};
                try {
                    if (metric == 0)
                        ProbeVmaf(depth, scene, fields, clip);
                    else
                        ProbeVmafV1(depth, scene, fields, clip);
                } catch (const std::exception& error) {
                    fields[3] = "-1";
                    fields[13] = error.what();
                    PrintCsvRow(fields);
                }
            }
        }
    }
    vqm::internal::ShutdownDefaultVulkanContext();
}
