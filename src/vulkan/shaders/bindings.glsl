// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// The storage interface every compute shader shares. Binding 0 is the arena as
// floats and binding 2 is the arena as words, including the per-image constant
// blocks and per-level table. Binding 3 is the current source image as words:
// the arena for internal inputs, or a borrowed VkBuffer for external inputs.
//
// Binding 1 is the arena's plane region, and its element type belongs to the
// metric, so each backend declares it here-adjacent rather than in this file:
// SSIMULACRA 2's ssimu2_common.glsl makes it float[] or float16_t under
// SS2_HALF_PLANES, Butteraugli's butter_common.glsl does not use it at all (its
// planes are f32 arena floats). The C++ mirrors live in the two backends.
//
// A shader includes its metric's common file after its #version directive;
// that file includes this one first, then declares its push block, then the
// helpers that use both.

layout(std430, binding = 0) buffer Arena { float arena[]; };

// The per-image constant block, read-only here: one descriptor per image of
// the pair, at pc.desc_base plus the word index color.glsl uses. The same
// words hold each metric's per-level table at pc.level_base.
layout(std430, binding = 2) readonly buffer Descs { uint desc[]; };

layout(std430, binding = 3) readonly buffer SourceWords { uint source_words[]; };
