// train_tokenizer — standalone byte-level BPE trainer.
//
// Reads a text column from a CSV, runs the BPE merge loop (see bpe_core.h)
// and writes tokenizer.json + tokenizer_vocabulary.csv. Pure C++, no Python.
//
// Usage:
//   train_tokenizer [--corpus P] [--column processed] [--output-dir D]
//                   [--steps N] [--min-count N] [--max-vocab-size N]
//                   [--special-tokens a,b,c] [--lowercase] [--quiet]

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "bpe_core.h"

namespace fs = std::filesystem;

static void usage(const char *prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "  --corpus PATH        input CSV (default: "
           "data/notebooks/claude_opus_4.6_4.7_reasoning_8.7k.csv)\n"
        << "  --column NAME        text column (default: processed)\n"
        << "  --output-dir DIR     output dir (default: data/tokenizer)\n"
        << "  --steps N            max merge steps (default: 100000)\n"
        << "  --min-count N        stop below this pair frequency (default: 5)\n"
        << "  --max-vocab-size N   hard vocab cap (default: 262144)\n"
        << "  --special-tokens A,B comma-separated specials (default: built-in)\n"
        << "  --lowercase          lowercase corpus first\n"
        << "  --quiet              less output\n";
}

static std::vector<std::string> split_comma(const std::string &s) {
    std::vector<std::string> o;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) o.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return o;
}

static std::string ascii_lower(std::string s) {
    for (char &c : s)
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return s;
}

int main(int argc, char **argv) {
    std::string corpus = "data/notebooks/claude_opus_4.6_4.7_reasoning_8.7k.csv";
    std::string column = "processed";
    std::string outdir = "data/tokenizer";
    bpe::TrainConfig cfg;
    std::vector<std::string> specials = {"<user>",     "</user>", "<assistant>",
                                         "</assistant>", "<system>", "</system>",
                                         "<think>",    "</think>"};
    bool lowercase = false, quiet = false;

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
        } else if (a == "--output-dir") {
            if (!need(v)) return 1;
            outdir = v;
        } else if (a == "--steps") {
            if (!need(v)) return 1;
            cfg.steps = std::stoi(v);
        } else if (a == "--min-count") {
            if (!need(v)) return 1;
            cfg.min_count = std::stoll(v);
        } else if (a == "--max-vocab-size") {
            if (!need(v)) return 1;
            cfg.max_vocab_size = std::stoll(v);
        } else if (a == "--special-tokens") {
            if (!need(v)) return 1;
            specials = split_comma(v);
        } else if (a == "--lowercase") {
            lowercase = true;
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
        if (!quiet)
            std::cerr << "Loading corpus from " << corpus << " (column='"
                      << column << "') ...\n";
        auto records = bpe::read_csv_records(corpus);
        auto rows = bpe::csv_column(records, column);
        std::string joined;
        size_t total_len = 0;
        for (auto &r : rows) total_len += r.size() + 1;
        joined.reserve(total_len);
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i) joined.push_back(' ');
            joined += rows[i];
        }
        if (lowercase) joined = ascii_lower(joined);
        if (!quiet)
            std::cerr << "Loaded corpus: " << joined.size() << " chars, "
                      << rows.size() << " rows (" << secs() << "s)\n";

        if (!quiet) std::cerr << "Pre-tokenizing + counting ...\n";
        auto toks = bpe::pre_tokenize(joined, specials);
        auto counted = bpe::count_words(toks);
        if (!quiet)
            std::cerr << "Distinct pre-tokens: " << counted.first.size()
                      << " (" << secs() << "s)\n";
        // Free the joined text early on big corpora.
        joined.clear();
        joined.shrink_to_fit();

        if (!quiet)
            std::cerr << "Training BPE: steps=" << cfg.steps
                      << " min_count=" << cfg.min_count
                      << " max_vocab=" << cfg.max_vocab_size << "\n";
        int report_every = std::max(1, cfg.steps / 1000);
        auto rules = bpe::train_bpe(
            counted.first, counted.second, specials, cfg,
            [&](const bpe::TrainProgress &pr) {
                if (quiet) return;
                if (pr.stopped) {
                    std::cerr << "\nStopping: " << pr.stop_reason << "\n";
                    return;
                }
                if (pr.step % report_every == 0 || pr.step == cfg.steps - 1) {
                    char buf[256];
                    snprintf(buf, sizeof buf,
                             "\r[%d/%d] rules=%d tok=%s freq=%lld (%.1fs)", pr.step,
                             cfg.steps, pr.rule_count,
                             pr.new_token_display.c_str(),
                             (long long)pr.frequency, secs());
                    std::cerr << buf << std::flush;
                }
            });
        if (!quiet) std::cerr << "\n";
        std::cout << "Learned " << rules.size() << " merge rules (" << secs()
                  << "s)\n";

        bpe::Vocab vocab = bpe::build_vocab(rules, specials);
        fs::create_directories(outdir);
        std::string json_path = (fs::path(outdir) / "tokenizer.json").string();
        std::string csv_path =
            (fs::path(outdir) / "tokenizer_vocabulary.csv").string();
        bpe::save_tokenizer_json(json_path, specials, rules, vocab.id_to_token);
        bpe::save_vocab_csv(csv_path, vocab.id_to_token);
        std::cout << "Saved: " << json_path << "\nSaved: " << csv_path
                  << "\nFinal vocab size: " << vocab.id_to_token.size() << "\n";

        // Round-trip sanity check.
        bpe::TokenizerData td;
        td.specials = specials;
        td.merges = rules;
        td.id_to_token = vocab.id_to_token;
        bpe::Encoder enc(td);
        std::string sample = "Hello, world!  Is this working?";
        auto ids = enc.encode_text(sample);
        std::string back = enc.decode(ids);
        std::cout << "\nRound-trip check:\n  in : '" << sample << "'\n  ids: [";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) std::cout << ",";
            std::cout << ids[i];
        }
        std::cout << "]\n  out: '" << back << "'\n";
        if (back != sample) {
            std::cerr << "round-trip MISMATCH\n";
            return 1;
        }
        std::cout << "  OK\n\nTotal time: " << secs() << "s\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
