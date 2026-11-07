// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <string_view>
#include <vector>

#include "support.h"

namespace vqm::test {
namespace {

std::vector<TestCase> g_tests;
int g_failures = 0;
std::string_view g_skip_reason;
bool g_missing_gpu = false;

void PrintName(const char* tag, std::string_view name) {
    std::fprintf(
        stdout, "%s %.*s\n", tag, static_cast<int>(name.size()), name.data()
    );
}

}  // namespace

void RegisterTest(std::string_view name, void (*run)()) {
    g_tests.push_back(TestCase{name, run});
}

void Fail(std::source_location where, std::string_view message) {
    std::fprintf(
        stderr, "%s:%u: FAILED: %.*s\n", where.file_name(), where.line(),
        static_cast<int>(message.size()), message.data()
    );
    ++g_failures;
}

void MarkSkip(std::string_view reason, bool missing_gpu) {
    g_skip_reason = reason;
    g_missing_gpu = g_missing_gpu || missing_gpu;
}

int RunAll(bool require_gpu) {
    std::size_t skipped = 0;
    for (const TestCase& test : g_tests) {
        PrintName("[ RUN      ]", test.name);
        const int failures_before = g_failures;
        g_skip_reason = {};
        try {
            test.run();
        } catch (const std::exception& e) {
            Fail(
                std::source_location::current(),
                std::format("unexpected exception: {}", e.what())
            );
        } catch (...) {
            Fail(
                std::source_location::current(),
                "unexpected non-standard exception"
            );
        }
        if (g_failures != failures_before) {
            PrintName("[  FAILED  ]", test.name);
        } else if (!g_skip_reason.empty()) {
            ++skipped;
            PrintName("[  SKIPPED ]", test.name);
            std::fprintf(
                stdout, "             %.*s\n",
                static_cast<int>(g_skip_reason.size()), g_skip_reason.data()
            );
        } else {
            PrintName("[       OK ]", test.name);
        }
    }
    std::fprintf(
        stdout, "%zu test(s), %d failed check(s)\n", g_tests.size(), g_failures
    );
    if (skipped != 0) std::fprintf(stdout, "%zu test(s) skipped\n", skipped);
    if (require_gpu && g_missing_gpu) {
        std::fprintf(
            stderr, "GPU required, but no usable Vulkan device was found\n"
        );
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    }
    if (g_failures != 0 || (require_gpu && g_missing_gpu)) {
        return EXIT_FAILURE;
    }
    std::printf(
        skipped == 0 ? "all tests passed\n" : "all executed tests passed\n"
    );
    return EXIT_SUCCESS;
}

}  // namespace vqm::test

namespace {

int RunTestMain(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--probe-gpu") {
        if (vqm::test::GpuAvailable()) return EXIT_SUCCESS;
        std::fprintf(
            stderr, "GPU required, but no usable Vulkan device was found\n"
        );
        return EXIT_FAILURE;
    }
    if (argc > 2 ||
        (argc == 2 && std::string_view(argv[1]) != "--require-gpu")) {
        std::fprintf(stderr, "usage: %s [--require-gpu|--probe-gpu]\n", argv[0]);
        return EXIT_FAILURE;
    }
    return vqm::test::RunAll(argc == 2);
}

}  // namespace

int main(int argc, char** argv) {
    int status = EXIT_FAILURE;
    try {
        status = RunTestMain(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test initialization: %s\n", error.what());
    }
    // Test-local pipelines and borrowed devices are gone before this point.
    // Explicitly release the retained device while driver and validation-layer
    // state is still alive: a second instance can register later exit cleanup
    // that runs before the singleton's fallback. Keep the owner-count check so
    // leaked pipeline owners fail the test process instead of hiding a leak.
    try {
        vqm::internal::ShutdownDefaultVulkanContext();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "test shutdown: %s\n", error.what());
        return EXIT_FAILURE;
    }
    return status;
}
