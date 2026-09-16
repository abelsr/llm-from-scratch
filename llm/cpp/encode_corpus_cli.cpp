// encode_corpus — standalone corpus encoder.
//
// Loads tokenizer.json, encodes a CSV text column row by row and writes the
// flat int32 (little-endian) id array to the cache path. Pure C++, no Python.
// Rows are encoded in parallel (each row is independent).
//
// Usage:
//   encode_corpus [--corpus P] [--column processed] [--tokenizer T.json]
//                 [--cache C.bin] [--force] [--max-rows N] [--jobs N] [--quiet]

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bpe_core.h"

namespace fs = std::filesystem;

static void show_progress(const char *label, size_t done, size_t total,
                          double elapsed, double rate = 0.0) {
    char buf[256];
    if (total == 0) {
        snprintf(buf, sizeof buf, "\r%-18s  (empty)  (%.1fs)", label,
                 elapsed);
    } else {
        double pct = 100.0 * (double)done / (double)total;
        if (rate > 0.0) {
            double eta = ((double)total - (double)done) / rate;
            snprintf(buf, sizeof buf,
                     "\r%-18s %6.2f%%  %zu/%zu  %.0f rows/s  ETA %.1fs",
                     label, pct, done, total, rate, eta);
        } else {
            snprintf(buf, sizeof buf, "\r%-18s %6.2f%%  %zu/%zu  (%.1fs)",
                     label, pct, done, total, elapsed);
        }
    }
    std::cerr << buf << std::flush;
}

static void usage(const char *prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --corpus PATH     input CSV (default: "
           "data/notebooks/claude_opus_4.6_4.7_reasoning_8.7k.csv)\n"
        << "  --column NAME     text column (default: processed)\n"
        << "  --tokenizer PATH  tokenizer.json (default: "
           "data/tokenizer/tokenizer.json)\n"
        << "  --cache PATH      output .bin (default: data/tokenizer/corpus_ids.bin)\n"
        << "  --force           re-encode even if cache exists\n"
        << "  --max-rows N      only encode the first N rows\n"
        << "  --jobs N          encoding threads (default: hardware concurrency)\n"
        << "  --quiet           no progress output\n";
}

int main(int argc, char **argv) {
    std::string corpus = "data/notebooks/claude_opus_4.6_4.7_reasoning_8.7k.csv";
    std::string column = "processed";
    std::string tok_path = "data/tokenizer/tokenizer.json";
    std::string cache = "data/tokenizer/corpus_ids.bin";
    bool force = false, quiet = false;
    long long max_rows = -1;
    int jobs = (int)std::thread::hardware_concurrency();
    if (jobs < 1) jobs = 1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](std::string &dst) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << a << "\n";
                return false;
            }
            dst = argv[++i];
            return true;
        };
        std::string v;
        if (a == "--corpus") {
            if (!need(corpus)) return 1;
        } else if (a == "--column") {
            if (!need(column)) return 1;
        } else if (a == "--tokenizer") {
            if (!need(tok_path)) return 1;
        } else if (a == "--cache") {
            if (!need(cache)) return 1;
        } else if (a == "--force") {
            force = true;
        } else if (a == "--max-rows") {
            if (!need(v)) return 1;
            max_rows = std::stoll(v);
        } else if (a == "--jobs") {
            if (!need(v)) return 1;
            jobs = std::stoi(v);
            if (jobs < 1) jobs = 1;
        } else if (a == "--quiet") {
            quiet = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "unknown arg: " << a << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t0)
            .count();
    };

    try {
        if (fs::exists(cache) && !force) {
            uintmax_t sz = fs::file_size(cache);
            std::cout << "Using cached encoded corpus at " << cache << " ("
                      << sz / 4 << " tokens)\n";
            return 0;
        }
        if (!quiet) std::cerr << "Loading tokenizer from " << tok_path << "\n";
        bpe::TokenizerData td = bpe::load_tokenizer_json(tok_path);
        if (!quiet)
            std::cerr << "  vocab_size=" << td.id_to_token.size()
                      << " merges=" << td.merges.size() << "\n";
        bpe::Encoder enc(td);

        if (!quiet)
            std::cerr << "Loading corpus rows from " << corpus << " ...\n";
        auto records = bpe::read_csv_records(
            corpus,
            [&](size_t done, size_t total) {
                if (!quiet) show_progress("Reading CSV", done, total, secs());
            },
            [&](size_t done, size_t total) {
                if (!quiet) show_progress("Parsing CSV", done, total, secs());
            });
        if (!quiet) std::cerr << "\n";
        auto rows = bpe::csv_column(records, column);
        if (max_rows >= 0 && (long long)rows.size() > max_rows)
            rows.resize((size_t)max_rows);
        size_t total = rows.size();
        if (!quiet) std::cerr << "  rows=" << total << "\n";

        std::vector<std::vector<int32_t>> per_row(total);
        std::atomic<size_t> done{0};
        std::atomic<bool> failed{false};
        std::string error;
        std::mutex err_mu, prog_mu;
        size_t report_every = std::max<size_t>(1, total / 50);

        auto worker = [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end && !failed.load(); ++i) {
                try {
                    per_row[i] = enc.encode_text(rows[i]);
                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (!failed.exchange(true)) error = e.what();
                    return;
                }
                size_t d = ++done;
                if (!quiet && (d % report_every == 0 || d == total)) {
                    std::lock_guard<std::mutex> lk(prog_mu);
                    double el = secs();
                    double rate = d / std::max(el, 1e-9);
                    show_progress("Encoding corpus", d, total, el, rate);
                }
            }
        };

        std::vector<std::thread> pool;
        size_t chunk = (total + (size_t)jobs - 1) / (size_t)jobs;
        for (int j = 0; j < jobs; ++j) {
            size_t b = (size_t)j * chunk;
            if (b >= total) break;
            pool.emplace_back(worker, b, std::min(total, b + chunk));
        }
        for (auto &t : pool) t.join();
        if (!quiet) std::cerr << "\n";
        if (failed.load()) {
            std::cerr << "ERROR: " << error << "\n";
            return 1;
        }

        size_t n_ids = 0;
        for (auto &r : per_row) n_ids += r.size();
        fs::path cp(cache);
        if (cp.has_parent_path()) fs::create_directories(cp.parent_path());
        std::ofstream out(cache, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write: " + cache);
        static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                      "int32 cache assumes little-endian");
        size_t written = 0;
        size_t last_reported = 0;
        size_t write_report_every = std::max<size_t>(1, n_ids / 50);
        for (auto &r : per_row) {
            out.write(reinterpret_cast<const char *>(r.data()),
                      (std::streamsize)(r.size() * sizeof(int32_t)));
            written += r.size();
            if (!quiet &&
                written != last_reported &&
                (written >= n_ids ||
                 written / write_report_every !=
                     last_reported / write_report_every)) {
                show_progress("Writing cache", written, n_ids, secs());
                last_reported = written;
            }
        }
        if (!quiet && n_ids == 0) show_progress("Writing cache", 0, 0, secs());
        out.close();
        if (!quiet) std::cerr << "\n";

        double dt = secs();
        std::cout << "Done in " << dt << "s. tokens=" << n_ids
                  << " cache=" << cache << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
