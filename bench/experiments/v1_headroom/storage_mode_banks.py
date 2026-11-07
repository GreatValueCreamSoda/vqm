#!/usr/bin/env python3
"""Install exact uint16 CAMBI mode banks in a fresh isolated source copy.

Only the horizontal and filtered banks change representation. Their arena
reservations, offsets and lifetimes stay unchanged, so this first arm prices
storage traffic without changing memory queries. No FP16 arithmetic is used.
The maintained checkout and its ancestors/descendants are refused deliberately.
See bench/notes/vmaf/research/v1-storage-dispatch-headroom.md.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMBI = Path("src/metrics/vmaf/cambi")


def replace_once(text: str, before: str, after: str) -> str:
    """Replace one exact source anchor.

    :param text: Source text to inspect or modify.
    :param before: Source anchor to find.
    :param after: Replacement text.
    :returns: Source text with the replacement applied.
    :raises ValueError: If the source anchor does not occur exactly once.
    """
    if text.count(before) != 1:
        raise ValueError(f"expected one unchanged source anchor: {before[:100]}")
    return text.replace(before, after)


def packed_shader(text: str) -> str:
    """Generate a shader variant with packed mode banks.

    :param text: Source text to inspect or modify.
    :returns: Shader source with the packed code buffer declaration.
    """
    text = replace_once(
        text, "#version 450", """#version 450
#extension GL_EXT_shader_16bit_storage : require"""
    )
    return replace_once(
        text, '#include "bindings.glsl"',
        """// Binding 2 aliases binding 0's arena at byte zero. Existing push offsets
// are 32-bit words; each code is instead addressed in the bank's first half.
// These banks are disjoint from every FP32 region and have one writer per
// 16-bit element. Typed storage avoids read/modify/write races on odd tails.
layout(std430, binding = 0) buffer Arena { float arena[]; };
layout(std430, binding = 2) buffer Codes { uint16_t codes[]; };"""
    )


def _patch_host(header: str, host: str) -> tuple[str, str]:
    """Generate declarations and dispatches for packed CAMBI mode banks.

    :param header: Original CAMBI declarations.
    :param host: Original CAMBI implementation.
    :returns: Updated CAMBI declarations and implementation.
    """
    header = replace_once(
        header, "    explicit VmafCambiDecimate(VulkanRuntime& runtime);",
        "    explicit VmafCambiDecimate(VulkanRuntime& runtime, bool allow_packed = false);"
    )
    header = replace_once(
        header, "    void Record(Size size, const VmafCambiDecimateBuffers& buffers);",
        """    void Record(Size size, const VmafCambiDecimateBuffers& buffers,
                bool packed_input = false);"""
    )
    # Target the private section adjacent to the next class, not other stages.
    header = replace_once(
        header, """    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

// ---- Mode filtering and local contrast ----""",
        """    std::array<VkPipeline, 2> pipelines_{};
};

// ---- Mode filtering and local contrast ----"""
    )
    header = replace_once(
        header,
        "    explicit VmafCambiContrast(VulkanRuntime& runtime, bool allow_half = true);",
        """    explicit VmafCambiContrast(VulkanRuntime& runtime, bool allow_half = true,
                               bool packed_banks = false);"""
    )
    header = replace_once(
        header, """    VulkanRuntime* runtime_;
    VmafCambiPrepare prepare_;""", """    VulkanRuntime* runtime_;
    bool packed_banks_ = false;
    VmafCambiPrepare prepare_;"""
    )
    header = replace_once(
        header, "// word. All memory must outlive GPU completion.",
        """// word. All memory must outlive GPU completion. In this experiment only,
// contrast_horizontal and filtered use uint16 storage when enabled; offsets
// and reservations remain 32-bit words, with the second half left unused."""
    )
    host = replace_once(
        host, """VmafCambiDecimate::VmafCambiDecimate(VulkanRuntime& runtime)
    : runtime_(&runtime),
      pipeline_(runtime.CreatePipeline("vmaf_cambi_decimate")) {}
VmafCambiDecimate::~VmafCambiDecimate() {
    runtime_->DestroyPipelines(std::array{pipeline_});
}""", """VmafCambiDecimate::VmafCambiDecimate(VulkanRuntime& runtime, bool allow_packed)
    : runtime_(&runtime),
      pipelines_(allow_packed && runtime.context().sixteen_bit_storage
                     ? runtime.CreatePipelines(std::array{
                           "vmaf_cambi_decimate", "vmaf_cambi_decimate_packed"})
                     : std::array<VkPipeline, 2>{
                           runtime.CreatePipeline("vmaf_cambi_decimate"),
                           VK_NULL_HANDLE}) {}
VmafCambiDecimate::~VmafCambiDecimate() {
    runtime_->DestroyPipelines(pipelines_);
}"""
    )
    host = replace_once(
        host,
        "void VmafCambiDecimate::Record(Size size, const VmafCambiDecimateBuffers& b) {",
        """void VmafCambiDecimate::Record(Size size, const VmafCambiDecimateBuffers& b,
                               bool packed_input) {
    if (packed_input && !pipelines_[1])
        throw std::logic_error("cambi decimate: packed pipeline unavailable");"""
    )
    host = replace_once(
        host, "        pipeline_, &push, sizeof(push), Groups(output.width, 16),",
        "        pipelines_[packed_input ? 1 : 0], &push, sizeof(push), Groups(output.width, 16),"
    )
    host = replace_once(
        host,
        "VmafCambiContrast::VmafCambiContrast(VulkanRuntime& runtime, bool allow_half)",
        """VmafCambiContrast::VmafCambiContrast(VulkanRuntime& runtime, bool allow_half,
                                   bool packed_banks)"""
    )
    host = replace_once(
        host,
        '          half_source_ ? "vmaf_cambi_contrast_half" : "vmaf_cambi_contrast"',
        """          packed_banks && runtime.context().sixteen_bit_storage
              ? (half_source_ ? "vmaf_cambi_contrast_packed_half"
                              : "vmaf_cambi_contrast_packed")
              : (half_source_ ? "vmaf_cambi_contrast_half" : "vmaf_cambi_contrast")"""
    )
    host = replace_once(
        host, """VmafCambi::VmafCambi(VulkanRuntime& runtime)
    : runtime_(&runtime),
      prepare_(runtime),
      decimate_(runtime),
      contrast_(runtime),""", """VmafCambi::VmafCambi(VulkanRuntime& runtime)
    : runtime_(&runtime),
      packed_banks_(runtime.context().sixteen_bit_storage),
      prepare_(runtime),
      decimate_(runtime, packed_banks_),
      contrast_(runtime, true, packed_banks_),"""
    )
    host = replace_once(
        host, "    std::uint32_t contrast_offset = b.contrast;",
        """    std::uint32_t contrast_offset = b.contrast;
    bool pixels_are_packed = false;"""
    )
    host = replace_once(
        host,
        "                {pixels_src, mask_src, b.scale_input, b.scale_mask[mask_bank]}",
        """                {pixels_src, mask_src, b.scale_input, b.scale_mask[mask_bank]},
                pixels_are_packed"""
    )
    host = replace_once(
        host, "        pixels_src = b.filtered;", """        pixels_src = b.filtered;
        pixels_are_packed = packed_banks_;"""
    )
    return header, host


def install(source: Path) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    source = source.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError(
            "use a fresh isolated source copy outside the maintained checkout"
        )
    header_path = source / CAMBI / "vmaf_cambi.h"
    host_path = source / CAMBI / "vmaf_cambi.cc"
    cmake_path = source / "CMakeLists.txt"
    contrast_path = source / CAMBI / "shaders/vmaf_cambi_contrast.comp"
    decimate_path = source / CAMBI / "shaders/vmaf_cambi_decimate.comp"
    generated_contrast = contrast_path.with_stem("vmaf_cambi_contrast_packed")
    generated_decimate = decimate_path.with_stem("vmaf_cambi_decimate_packed")
    if generated_contrast.exists() or generated_decimate.exists():
        raise ValueError("packed shaders already exist; use a fresh source copy")
    header, host, cmake = (p.read_text() for p in (header_path, host_path, cmake_path))
    contrast = packed_shader(contrast_path.read_text())
    decimate = packed_shader(decimate_path.read_text())
    contrast = replace_once(
        contrast, "pc;\n", """pc;

float load_horizontal(uint index) {
    return float(uint(codes[2u * pc.horizontal + index]));
}
void store_horizontal(uint index, float value) {
    codes[2u * pc.horizontal + index] = uint16_t(uint(value));
}
float load_filtered(uint index) {
    return float(uint(codes[2u * pc.filtered + index]));
}
void store_filtered(uint index, float value) {
    codes[2u * pc.filtered + index] = uint16_t(uint(value));
}
"""
    )
    contrast = replace_once(
        contrast, "    int value = int(arena[pc.filtered + at]);",
        "    int value = int(load_filtered(at));"
    )
    contrast = replace_once(
        contrast, "            arena[pc.horizontal + index] = center;",
        "            store_horizontal(index, center);"
    )
    contrast = replace_once(
        contrast, """            arena[pc.horizontal + index] = mode3(
                arena[pc.input_offset + index - 1u], center,
                arena[pc.input_offset + index + 1u]
            );""", """            store_horizontal(index, mode3(
                arena[pc.input_offset + index - 1u], center,
                arena[pc.input_offset + index + 1u]
            ));"""
    )
    contrast = replace_once(
        contrast,
        "        arena[pc.filtered + index] = arena[pc.input_offset + index];",
        "        store_filtered(index, arena[pc.input_offset + index]);"
    )
    contrast = replace_once(
        contrast, """        arena[pc.filtered + index] = mode3(
            arena[pc.horizontal + index - pc.width],
            arena[pc.horizontal + index], arena[pc.horizontal + index + pc.width]
        );""", """        store_filtered(index, mode3(
            load_horizontal(index - pc.width),
            load_horizontal(index), load_horizontal(index + pc.width)
        ));"""
    )
    decimate = replace_once(
        decimate, "    arena[pc.pixels_out + target] = arena[pc.pixels + source];",
        "    arena[pc.pixels_out + target] = float(uint(codes[2u * pc.pixels + source]));"
    )

    header, host = _patch_host(header, host)
    cmake = replace_once(
        cmake,
        "vqm_add_shader(vmaf_cambi_decimate metrics/vmaf/cambi/shaders/vmaf_cambi_decimate.comp)",
        """vqm_add_shader(vmaf_cambi_decimate metrics/vmaf/cambi/shaders/vmaf_cambi_decimate.comp)
vqm_add_shader(vmaf_cambi_contrast_packed metrics/vmaf/cambi/shaders/vmaf_cambi_contrast_packed.comp)
vqm_add_shader(vmaf_cambi_contrast_packed_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast_packed.comp -DVQM_CAMBI_HALF_SOURCE=1)
vqm_add_shader(vmaf_cambi_decimate_packed metrics/vmaf/cambi/shaders/vmaf_cambi_decimate_packed.comp)"""
    )
    # Validate every fragile anchor before any source file is written.
    for path, text in ((header_path, header), (host_path, host), (cmake_path, cmake),
                       (generated_contrast, contrast), (generated_decimate, decimate)):
        path.write_text(text)


def main() -> None:
    """Install the packed CAMBI mode-bank experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    args = parser.parse_args()
    install(args.source_dir)
    print(
        "Installed uint16 CAMBI horizontal/filtered banks with unchanged reservations."
    )


if __name__ == "__main__":
    main()
