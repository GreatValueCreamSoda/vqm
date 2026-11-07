// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.

// Decode-ahead queue for a batch scorer: a small pool of threads decodes pairs
// in input order while the consumer scores them on its own thread, and the
// results are handed over in input order. The point is a GPU scorer with one
// pipeline and one device: decoding one pair at a time leaves the device idle
// between calls, and spawning one process per pair only hides that behind
// startup cost.
//
// The queue is bounded, so a corpus of thousands of pairs does not accumulate
// in memory. A pair whose decode fails is still delivered in its place, with
// the loader's error, so the one-line-per-pair contract holds.

#ifndef VQM_BENCH_INMEMORY_PREFETCH_H_
#define VQM_BENCH_INMEMORY_PREFETCH_H_

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pairs.h"

namespace inmem {

template <typename Item>
class Prefetch {
public:
    using Loader = std::function<bool(const Pair&, Item&, std::string&)>;

    // `threads` decode workers, at most `depth` results buffered ahead.
    Prefetch(const std::vector<Pair>& pairs, int threads, Loader loader)
        : pairs_(pairs),
          loader_(std::move(loader)),
          depth_(static_cast<std::size_t>(std::max(1, threads)) * 2 + 1) {
        for (int index = 0; index < std::max(1, threads); ++index) {
            workers_.emplace_back([this] { Work(); });
        }
    }

    ~Prefetch() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }

    Prefetch(const Prefetch&) = delete;
    Prefetch& operator=(const Prefetch&) = delete;

    // One call per pair, in input order. Returns false with `error` set when
    // that pair could not be decoded; after the last pair it returns false
    // with an empty error. `index` receives the delivered pair's position in
    // the input list -- on a decode failure too, so the caller can put the
    // error in that pair's slot -- and is untouched at the end of the list.
    bool Take(Item& item, std::size_t* index, std::string* error) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] {
            return stop_ || buffered_.count(deliver_) != 0 ||
                   deliver_ >= pairs_.size();
        });
        if (deliver_ >= pairs_.size()) {
            error->clear();
            return false;
        }
        const auto found = buffered_.find(deliver_);
        if (found == buffered_.end()) {
            error->clear();
            return false;
        }
        if (index != nullptr) *index = deliver_;
        Result result = std::move(found->second);
        buffered_.erase(found);
        ++deliver_;
        lock.unlock();
        // Both variables take notify_all, not notify_one: several consumers
        // wait on `ready_`, each for whichever index `deliver_` holds, so a
        // single wake-up can land on a consumer whose index is still missing
        // and be spent. That is a lost wakeup as soon as the decoders are done
        // and no further notification comes -- measured: four consumers with
        // sixteen decode threads deadlocked, one consumer never did. The
        // waiter counts here are a handful of threads.
        wake_.notify_all();
        ready_.notify_all();
        if (!result.ok) {
            *error = std::move(result.error);
            return false;
        }
        item = std::move(result.item);
        error->clear();
        return true;
    }

    bool Take(Item& item, std::string* error) {
        return Take(item, nullptr, error);
    }

private:
    struct Result {
        Item item{};
        std::string error;
        bool ok = false;
    };

    void Work() {
        while (true) {
            std::size_t index = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] {
                    return stop_ ||
                           (next_ < pairs_.size() && buffered_.size() < depth_);
                });
                if (stop_ || next_ >= pairs_.size()) return;
                index = next_++;
            }
            Result result;
            result.ok = loader_(pairs_[index], result.item, result.error);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stop_) return;
                buffered_.emplace(index, std::move(result));
            }
            ready_.notify_all();
        }
    }

    const std::vector<Pair>& pairs_;
    Loader loader_;
    const std::size_t depth_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable wake_;
    std::map<std::size_t, Result> buffered_;
    std::size_t next_ = 0;
    std::size_t deliver_ = 0;
    bool stop_ = false;
    std::vector<std::thread> workers_;
};

}  // namespace inmem

#endif  // VQM_BENCH_INMEMORY_PREFETCH_H_
