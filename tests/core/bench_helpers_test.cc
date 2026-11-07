// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// CPU-only checks of benchmark input validation and throughput accounting.
#include <atomic>
#include <limits>
#include <string>
#include <vector>

#include "inmemory/driver.h"

namespace {

struct Item {
    int width = 0;
    int height = 0;
    double decode_ms = 0.0;
};

}  // namespace

int main() {
    if (inmem::CsvField("plain.png") != "plain.png" ||
        inmem::CsvField("a,b\"c\n.png") != "\"a,b\"\"c\n.png\"") {
        return 1;
    }
    if (!inmem::ValidWindowSeconds(1.0) || inmem::ValidWindowSeconds(0.0) ||
        inmem::ValidWindowSeconds(std::numeric_limits<double>::infinity()) ||
        inmem::ValidWindowSeconds(std::numeric_limits<double>::quiet_NaN()) ||
        inmem::ValidWindowSeconds(1e300)) {
        return 1;
    }

    const std::vector<inmem::Pair> pairs{{"one", "a"}, {"two", "b"}};
    const auto load = [](const inmem::Pair& pair, Item& item, std::string&) {
        item.width = pair.original == "one" ? 32 : 64;
        item.height = 32;
        return true;
    };
    std::atomic<int> calls{0};
    const auto score =
        [&calls](Item&, std::size_t, int, double*, double*, std::string&) {
            ++calls;
            return true;
        };
    inmem::ThroughputStats stats;
    std::string error;
    if (inmem::RunThroughput<Item>(
            pairs, 2, 2, 0.01, load, score, &stats, &error
        ) ||
        error.find("same geometry") == std::string::npos || calls != 0) {
        return 1;
    }
    return 0;
}
