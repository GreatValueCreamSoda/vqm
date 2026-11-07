// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// The tab-separated pair list every in-memory harness and the benchmark
// runners share: one "original<TAB>distorted" line per pair, '#' comments and
// blank lines skipped.

#ifndef VQM_BENCH_INMEMORY_PAIRS_H_
#define VQM_BENCH_INMEMORY_PAIRS_H_

#include <format>
#include <fstream>
#include <string>
#include <vector>

namespace inmem {

struct Pair {
    std::string original;
    std::string distorted;
};

inline bool ReadPairsFile(
    const std::string& path, std::vector<Pair>& pairs, std::string* error,
    bool include_path_in_error = true
) {
    std::ifstream file(path);
    if (!file) {
        *error = std::format("could not open pairs file {}", path);
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '#') continue;
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            *error = include_path_in_error
                         ? std::format(
                               "{}: line has no tab separator: {}", path, line
                           )
                         : std::format(
                               "pairs file line has no tab separator: {}", line
                           );
            return false;
        }
        pairs.push_back(Pair{line.substr(0, tab), line.substr(tab + 1)});
    }
    return true;
}

}  // namespace inmem

#endif  // VQM_BENCH_INMEMORY_PAIRS_H_
