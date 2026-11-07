#!/usr/bin/env python3
"""Install independent exact CAMBI contrast experiments in an isolated source copy.

No maintained source is edited. Examples:
  python3 contrast_variants.py /tmp/v1-mode --mode-only
  python3 contrast_variants.py /tmp/v1-radius --radius8
  python3 contrast_variants.py /tmp/v1-wide --tile-width 32

Mode separation compiles the existing shader a second time without reachable
contrast code. Radius specialization keeps the original pipeline for windows
above 17. Geometry experiments affect only enabled FP16 source storage; the
portable uint fallback retains its original 16x8 geometry.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAMBI = Path("src/metrics/vmaf/cambi")
SHADER = CAMBI / "shaders/vmaf_cambi_contrast.comp"


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


def parameterize(shader: str, mode_only: bool, reuse_tail: bool = False) -> str:
    """Parameterize the contrast shader for experimental tile geometry.

    :param shader: Original GLSL source text.
    :param mode_only: Whether to change only the mode-filter path.
    :param reuse_tail: Whether to reuse the tail of shared storage.
    :returns: Shader source with the selected compile-time options.
    :raises ValueError: If an expected shader source anchor is missing.
    """
    shader = replace_once(
        shader, "layout(local_size_x = 16, local_size_y = 16) in;",
        """#ifndef VQM_CAMBI_TILE_INVOCATIONS
#define VQM_CAMBI_TILE_INVOCATIONS 256
#endif
layout(local_size_x = 16, local_size_y = VQM_CAMBI_TILE_INVOCATIONS / 16) in;
#ifndef VQM_CAMBI_TILE_WIDTH
#define VQM_CAMBI_TILE_WIDTH 16
#endif
#ifndef VQM_CAMBI_TILE_ROWS
#define VQM_CAMBI_TILE_ROWS 8
#endif
#ifndef VQM_CAMBI_MAX_RADIUS
#define VQM_CAMBI_MAX_RADIUS 16
#endif
const uint TILE_WIDTH = VQM_CAMBI_TILE_WIDTH;
const uint TILE_ROWS = VQM_CAMBI_TILE_ROWS;
const uint LOCAL_INVOCATIONS = VQM_CAMBI_TILE_INVOCATIONS;
const uint ROW_LANES = LOCAL_INVOCATIONS / TILE_ROWS;
const uint ALL_ROWS = (1u << TILE_ROWS) - 1u;"""
    )
    begin = shader.index("// Eight output rows share")
    end = shader.index("layout(push_constant)", begin)
    shader = shader[:begin
                    ] + """// Experimental fully buffered tile. Integer histogram and FP32 contrast
// arithmetic are unchanged; radius specialization changes capacity only.
shared CAMBI_SOURCE_TYPE source_tile[
    (VQM_CAMBI_TILE_WIDTH + 2 * VQM_CAMBI_MAX_RADIUS) *
    (VQM_CAMBI_TILE_ROWS + 2 * VQM_CAMBI_MAX_RADIUS)];
shared uint histograms[VQM_CAMBI_TILE_ROWS * 256];
shared uint active_rows;
shared float contrast_terms[VQM_CAMBI_TILE_ROWS * 4];
""" + shader[end:]
    replacements = [
        ("gl_LocalInvocationIndex % 32u", "gl_LocalInvocationIndex % ROW_LANES"),
        ("gl_LocalInvocationIndex / 32u", "gl_LocalInvocationIndex / ROW_LANES"),
        ("16u + 2u * pc.radius", "TILE_WIDTH + 2u * pc.radius"),
        ("gl_WorkGroupID.x * 16u", "gl_WorkGroupID.x * TILE_WIDTH"),
        ("gl_WorkGroupID.y * 8u", "gl_WorkGroupID.y * TILE_ROWS"),
        (
            "if (lane < 16u) {",
            "for (uint center_x = lane; center_x < TILE_WIDTH; center_x += ROW_LANES) {"
        ), ("* TILE_WIDTH + lane", "* TILE_WIDTH + center_x"),
        ("* extent + lane + pc.radius", "* extent + center_x + pc.radius"),
        ("i < 8u * 256u", "i < TILE_ROWS * 256u"),
        ("extent * (8u + 2u * pc.radius)", "extent * (TILE_ROWS + 2u * pc.radius)"),
        ("pc.radius + 16u", "pc.radius + TILE_WIDTH"),
        ("pc.radius + 8u", "pc.radius + TILE_ROWS"),
        ("active_rows == 255u", "active_rows == ALL_ROWS"),
        ("target < 8u", "target < TILE_ROWS"), ("i += 32u", "i += ROW_LANES"),
        ("x < 16u", "x < TILE_WIDTH"), ("x == 15u", "x + 1u == TILE_WIDTH"),
        ("k += 32u", "k += ROW_LANES"),
        (
            "// The 16x8 output center rectangle is already in the shared tile.",
            "// The output center rectangle is already in the shared tile."
        ),
    ]
    for before, after in replacements:
        if before not in shader:
            raise ValueError(f"missing shader geometry anchor: {before}")
        shader = shader.replace(before, after)
    shader = replace_once(
        shader, """            histograms[target * 256u + gl_LocalInvocationIndex] =
                histograms[(target - 1u) * 256u + gl_LocalInvocationIndex];""",
        """            for (uint i = gl_LocalInvocationIndex; i < 256u; i += LOCAL_INVOCATIONS)
                histograms[target * 256u + i] = histograms[(target - 1u) * 256u + i];"""
    )
    # Histogram row stride remains256 words. Only cooperative work scheduling
    # follows the selected physical invocation count.
    shader = shader.replace("i += 256u", "i += LOCAL_INVOCATIONS")
    shader = shader.replace("x += 256u", "x += LOCAL_INVOCATIONS")
    if reuse_tail:
        shader = replace_once(
            shader, "shared uint active_rows;\n",
            "#define CAMBI_ACTIVE_ROWS histograms[TILE_ROWS * 256u - 1u]\n"
        )
        shader = replace_once(
            shader, "shared float contrast_terms[VQM_CAMBI_TILE_ROWS * 4];\n", ""
        )
        shader = shader.replace("active_rows", "CAMBI_ACTIVE_ROWS")
        shader = replace_once(
            shader, """    barrier();
    if (CAMBI_ACTIVE_ROWS == 0u) {""", """    barrier();
    uint active_mask = CAMBI_ACTIVE_ROWS;
    if (active_mask == 0u) {"""
        )
        shader = replace_once(
            shader, "        histograms[i] = 0u;",
            """        // Keep the flag alive until every invocation has cached it. The
        // existing halo barrier precedes any clone that can overwrite it.
        if (i + 1u != TILE_ROWS * 256u) histograms[i] = 0u;"""
        )
        shader = shader.replace(
            "(CAMBI_ACTIVE_ROWS & (1u << row))", "(active_mask & (1u << row))"
        )
        shader = shader.replace(
            "CAMBI_ACTIVE_ROWS == ALL_ROWS", "active_mask == ALL_ROWS"
        )
        shader = replace_once(
            shader, "            contrast_terms[row * 4u + lane] = result;",
            "            histograms[row * 256u + 249u + lane] = floatBitsToUint(result);"
        )
        shader = replace_once(
            shader, "float result = contrast_terms[row * 4u];",
            "float result = uintBitsToFloat(histograms[row * 256u + 249u]);"
        )
        shader = replace_once(
            shader, "result = max(result, contrast_terms[row * 4u + d]);",
            "result = max(result, uintBitsToFloat(histograms[row * 256u + 249u + d]));"
        )
    if mode_only:
        shader = replace_once(
            shader, """    if (pc.phase == 2u) {
        contrast_map();
        return;
    }""", """#ifndef VQM_CAMBI_MODE_ONLY
    if (pc.phase == 2u) {
        contrast_map();
        return;
    }
#endif"""
        )
    return shader


def _validate_install(
    source: Path, mode_only: bool, radius8: bool, width: int, rows: int,
    reuse_tail: bool, invocations: int, small_window_only: bool
) -> Path:
    """Resolve the source path and reject incompatible experiment selections.

    :param source: Target source copy.
    :param mode_only: Whether to split the mode pass into its own shader.
    :param radius8: Whether to add radius-eight specialization.
    :param width: Workgroup output columns.
    :param rows: Workgroup output rows.
    :param reuse_tail: Whether to reuse the histogram tail.
    :param invocations: Workgroup invocation count.
    :param small_window_only: Whether to use a dedicated small-window shader.
    :returns: Resolved path of the isolated source copy.
    :raises ValueError: If the target or combination is unsupported.
    """
    source = source.resolve()
    if source == ROOT or source.is_relative_to(ROOT) or ROOT.is_relative_to(source):
        raise ValueError(
            "use a fresh isolated source copy outside the maintained checkout"
        )
    if not mode_only and not radius8 and not reuse_tail and (width, rows) == (16, 8):
        raise ValueError("select at least one experimental arm")
    if invocations not in (128, 256) or (invocations == 128 and
                                         (rows != 4 or not mode_only)):
        raise ValueError("128 invocations requires --tile-rows 4 and --mode-only")
    if small_window_only and (not mode_only or invocations != 128 or rows != 4):
        raise ValueError(
            "--small-window-only requires --mode-only --tile-rows 4 --tile-invocations 128"
        )
    if width not in (16, 32) or rows not in (4, 8):
        raise ValueError("supported experimental geometry is width 16/32, rows 4/8")
    return source


def _shader_rules(
    cmake: str, mode_only: bool, radius8: bool, width: int, rows: int, invocations: int,
    small_window_only: bool
) -> str:
    """Generate the shader build rules for the selected geometry.

    :param cmake: Original build rules.
    :param mode_only: Whether to add a mode-only module.
    :param radius8: Whether to specialize radius-eight modules.
    :param width: Workgroup output columns.
    :param rows: Workgroup output rows.
    :param invocations: Workgroup invocation count.
    :param small_window_only: Whether to use a dedicated small-window module.
    :returns: Build rules with the new variants.
    """
    half_rule = "vqm_add_shader(vmaf_cambi_contrast_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_HALF_SOURCE=1)"
    geometry = "" if (width, rows) == (
        16, 8
    ) else f" -DVQM_CAMBI_TILE_WIDTH={width} -DVQM_CAMBI_TILE_ROWS={rows}"
    if invocations != 256:
        geometry += f" -DVQM_CAMBI_TILE_INVOCATIONS={invocations}"
    rules = half_rule[:-1] + geometry + ")"
    if small_window_only:
        rules = half_rule
        capacity = " -DVQM_CAMBI_MAX_RADIUS=8" if radius8 else ""
        rules += "\nvqm_add_shader(vmaf_cambi_contrast_small_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_HALF_SOURCE=1" + geometry + capacity + ")"
    elif radius8:
        rules += "\nvqm_add_shader(vmaf_cambi_contrast_radius8 metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_MAX_RADIUS=8)"
        rules += "\nvqm_add_shader(vmaf_cambi_contrast_radius8_half metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_HALF_SOURCE=1 -DVQM_CAMBI_MAX_RADIUS=8" + geometry + ")"
    if mode_only:
        rules += "\nvqm_add_shader(vmaf_cambi_mode metrics/vmaf/cambi/shaders/vmaf_cambi_contrast.comp -DVQM_CAMBI_MODE_ONLY=1)"
    cmake = replace_once(cmake, half_rule, rules)
    return cmake


def _contrast_dispatch(
    host: str, header: str, mode_only: bool, radius8: bool, width: int, rows: int,
    small_window_only: bool
) -> tuple[str, str]:
    """Generate pipeline ownership and dispatch selection for the chosen shader set.

    :param host: Original CAMBI implementation.
    :param header: Original CAMBI declarations.
    :param mode_only: Whether to dispatch a separate mode shader.
    :param radius8: Whether to select a radius-eight shader when possible.
    :param width: Workgroup output columns.
    :param rows: Workgroup output rows.
    :param small_window_only: Whether to specialize only small windows.
    :returns: Updated host source and declarations.
    """
    start = host.index("VmafCambiContrast::VmafCambiContrast(")
    end = host.index("std::size_t VmafCambiContrast::Pixels", start)
    names = ['half_source_ ? "vmaf_cambi_contrast_half" : "vmaf_cambi_contrast"']
    if small_window_only:
        names.append(
            'half_source_ ? "vmaf_cambi_contrast_small_half" : "vmaf_cambi_contrast"'
        )
    elif radius8:
        names.append(
            'half_source_ ? "vmaf_cambi_contrast_radius8_half" : "vmaf_cambi_contrast_radius8"'
        )
    if mode_only:
        names.append('"vmaf_cambi_mode"')
    # Every half geometry here fits the old 16,004-byte portable minimum.
    constructor = """VmafCambiContrast::VmafCambiContrast(VulkanRuntime& runtime, bool allow_half)
    : runtime_(&runtime),
      half_source_(allow_half && runtime.context().shader_float16) {
    const std::array names{
        """ + ",\n        ".join(
        names
    ) + """};
    pipelines_ = runtime_->CreatePipelines(names);
}
VmafCambiContrast::~VmafCambiContrast() {
    runtime_->DestroyPipelines(pipelines_);
}
"""
    host = host[:start] + constructor + host[end:]
    start = header.index("class VmafCambiContrast {")
    end = header.index("// ---- Exact top-k pooling ----", start)
    section = replace_once(
        header[start:end], "VkPipeline pipeline_ = VK_NULL_HANDLE;",
        f"std::array<VkPipeline, {len(names)}> pipelines_{{}};"
    )
    header = header[:start] + section + header[end:]
    start = host.index("void VmafCambiContrast::Record(")
    end = host.index("// ---- Exact top-k pooling ----", start)
    record = host[start:end]
    if small_window_only:
        selection = f"""    const auto& limits = runtime_->context().limits;
    const bool small_window = half_source_ && window <= 17 &&
        Groups(size.height, {rows}) <= limits.maxComputeWorkGroupCount[1];
    const auto contrast_pipeline = small_window ? pipelines_[1] : pipelines_[0];
    const unsigned tile_width = small_window ? {width}u : 16u;
    const unsigned tile_rows = small_window ? {rows}u : 8u;"""
    else:
        selection = f"""    const auto contrast_pipeline = {'window <= 17 ? pipelines_[1] : pipelines_[0]' if radius8 else 'pipelines_[0]'};
    const unsigned tile_width = half_source_ ? {width}u : 16u;
    const unsigned tile_rows = half_source_ ? {rows}u : 8u;
    const auto& limits = runtime_->context().limits;"""
    record = replace_once(
        record, "    const auto& limits = runtime_->context().limits;", selection
    )
    record = replace_once(
        record, """Groups(size.width, 16) > limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height, 8) > limits.maxComputeWorkGroupCount[1]""",
        """std::max(Groups(size.width, 16), Groups(size.width, tile_width)) >
            limits.maxComputeWorkGroupCount[0] ||
        Groups(size.height, tile_rows) > limits.maxComputeWorkGroupCount[1]"""
    )
    mode_pipeline = f"pipelines_[{len(names) - 1}]" if mode_only else "contrast_pipeline"
    record = replace_once(
        record,
        """            pipeline_, &push, sizeof(push), Groups(size.width, 16),""",
        f"""            {mode_pipeline}, &push, sizeof(push), Groups(size.width, 16),"""
    )
    record = replace_once(
        record,
        """    // Sliding histograms bound each workgroup to 16x8 outputs and reuse
    // neighboring windows. Distinct groups write distinct output pixels.""",
        """    // Experimental geometry changes only the contrast dispatch. Mode passes
    // retain 16x16 coverage; contrast groups still own distinct output pixels."""
    )
    record = replace_once(
        record, """        pipeline_, &push, sizeof(push), Groups(size.width, 16),
        Groups(size.height, 8), 1""",
        """        contrast_pipeline, &push, sizeof(push), Groups(size.width, tile_width),
        Groups(size.height, tile_rows), 1"""
    )
    host = host[:start] + record + host[end:]
    return host, header


def install(
    source: Path, *, mode_only: bool = False, radius8: bool = False, width: int = 16,
    rows: int = 8, reuse_tail: bool = False, invocations: int = 256,
    small_window_only: bool = False
) -> None:
    """Install the research variant in an isolated source copy.

    :param source: Directory containing the isolated source copy.
    :param mode_only: Whether to change only the mode-filter path.
    :param radius8: Whether to cap the supported radius at eight.
    :param width: Width in samples or workgroup output columns.
    :param rows: Workgroup output row count.
    :param reuse_tail: Whether to reuse the tail of shared storage.
    :param invocations: Number of workgroup invocations.
    :param small_window_only: Whether to restrict specialization to small windows.
    :raises ValueError: If the target or expected source layout is unsuitable for the
        variant.
    """
    source = _validate_install(
        source, mode_only, radius8, width, rows, reuse_tail, invocations,
        small_window_only
    )
    paths = [
        SHADER, CAMBI / "vmaf_cambi.cc", CAMBI / "vmaf_cambi.h",
        Path("CMakeLists.txt")
    ]
    texts = {path: (source / path).read_text() for path in paths}
    shader, host, header, cmake = (texts[path] for path in paths)
    shader = parameterize(shader, mode_only, reuse_tail)
    if reuse_tail and not mode_only and not radius8 and (width, rows) == (16, 8):
        # This independent arm changes only shader storage; preserve host code,
        # pipeline ownership and CMake byte-for-byte.
        (source / SHADER).write_text(shader)
        return
    cmake = _shader_rules(
        cmake, mode_only, radius8, width, rows, invocations, small_window_only
    )
    host, header = _contrast_dispatch(
        host, header, mode_only, radius8, width, rows, small_window_only
    )
    # The experimental 4-row grid needs a correspondingly conservative planner.
    if rows == 4 and not small_window_only:
        planner = Path("src/metrics/vmaf/v1/vmaf_v1_pipeline.cc")
        plan_text = (source / planner).read_text()
        plan_text = replace_once(
            plan_text, "Groups(cambi.max_scale_size.height, 8)",
            "Groups(cambi.max_scale_size.height, 4)"
        )
        texts[planner] = plan_text
    texts.update(
        {
            SHADER: shader,
            CAMBI / "vmaf_cambi.cc": host,
            CAMBI / "vmaf_cambi.h": header,
            Path("CMakeLists.txt"): cmake
        }
    )
    # Validate every anchor before any write. Fresh copies are deliberately required.
    for path, content in texts.items():
        (source / path).write_text(content)


def main() -> None:
    """Install the requested CAMBI contrast geometry and shader variants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--mode-only", action="store_true")
    parser.add_argument("--radius8", action="store_true")
    parser.add_argument("--reuse-tail", action="store_true")
    parser.add_argument("--tile-width", type=int, choices=(16, 32), default=16)
    parser.add_argument("--tile-rows", type=int, choices=(4, 8), default=8)
    parser.add_argument("--tile-invocations", type=int, choices=(128, 256), default=256)
    parser.add_argument("--small-window-only", action="store_true")
    args = parser.parse_args()
    install(
        args.source_dir, mode_only=args.mode_only, radius8=args.radius8,
        width=args.tile_width, rows=args.tile_rows, reuse_tail=args.reuse_tail,
        invocations=args.tile_invocations, small_window_only=args.small_window_only
    )
    print(
        "Installed isolated CAMBI contrast experiment; run correctness and interleaved speed gates."
    )


if __name__ == "__main__":
    main()
