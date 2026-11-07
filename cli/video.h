// Copyright (c) 2026 VQM contributors. BSD-3-Clause; see LICENSE.

#pragma once

#include "options.h"

namespace vqm::cli {

// Detect Y4M content after legacy still-image validation, independent of suffix.
[[nodiscard]] bool HasY4mInput(const Options& options);

// Handle compressed video before the legacy still-image loader. Returns false
// for other inputs, including all inputs in a build without NVDEC.
[[nodiscard]] bool TryRunNvdecPair(const Options& options);

// Process complete ordered inputs, with one result per pair and metric.
// A failed pair throws before any of its results are printed.
int RunSequences(const Options& options);

}  // namespace vqm::cli
