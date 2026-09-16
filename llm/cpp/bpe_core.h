// bpe_core.h — shared core for the standalone BPE tools.
//
// Used by train_tokenizer_cli.cpp and encode_corpus_cli.cpp. Header-only so
// the tools build with a plain `g++ -O3` (no third-party dependencies).
//
// Semantics mirror the pure-Python implementation:
//   * pre-tokenization: split on special tokens, then `\s+\S+|\S+|\s+`
//     (ASCII whitespace; bytes >= 0x80 count as non-space),
//   * training: base-split to bytes, inverted-index merge loop with the same
//     stop rules (min_count, max_vocab_size),
//   * encoding: replay merges in rank order, specials never merge,
//   * tokenizer.json: {"version":1,"special_tokens":[...],
//     "merges":[[tok,tok],...],"id_to_token":[tok,...]} with
//     tok = {"type":"str","text":...} | {"type":"bytes","hex":...},
//   * CSV: RFC4180-ish reader (quotes, doubled quotes, embedded newlines)
//     matching what Python's csv module yields for our corpora.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace bpe {

using Id = int32_t;      // token id inside the merge/content tables
using OutId = int32_t;   // final vocabulary id (written to disk)
using PairKey = int64_t; // (Id_a << 32) | Id_b
using Count = long long;

static inline PairKey pack_pair(Id a, Id b) {
    return (static_cast<PairKey>(static_cast<uint32_t>(a)) << 32) |
           static_cast<PairKey>(static_cast<uint32_t>(b));
}
static inline Id pair_first(PairKey p) {
    return static_cast<Id>(static_cast<uint32_t>(p >> 32));
}
static inline Id pair_second(PairKey p) {
    return static_cast<Id>(static_cast<uint32_t>(p & 0xffffffffLL));
}

static inline bool is_space_byte(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

static std::string hex_of(const std::string &s) {
    static const char *d = "0123456789abcdef";
    std::string o;
    o.reserve(s.size() * 2);
    for (unsigned char c : s) {
        o.push_back(d[c >> 4]);
        o.push_back(d[c & 15]);
    }
    return o;
}

static std::string bytes_of_hex(const std::string &h) {
    if (h.size() % 2 != 0) throw std::runtime_error("odd-length hex string");
    auto val = [](char c) -> unsigned {
        if (c >= '0' && c <= '9') return (unsigned)(c - '0');
        if (c >= 'a' && c <= 'f') return (unsigned)(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return (unsigned)(c - 'A' + 10);
        throw std::runtime_error("invalid hex digit");
    };
    std::string o;
    o.reserve(h.size() / 2);
    for (size_t i = 0; i < h.size(); i += 2)
        o.push_back((char)((val(h[i]) << 4) | val(h[i + 1])));
    return o;
}

// Best-effort UTF-8 validation (for human-readable token display).
static bool valid_utf8(const std::string &s) {
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        size_t need = 0;
        if (c < 0x80) {
            ++i;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            need = 1;
            if (c < 0xC2) return false;
        } else if ((c & 0xF0) == 0xE0) {
            need = 2;
        } else if ((c & 0xF8) == 0xF0) {
            need = 3;
            if (c > 0xF4) return false;
        } else {
            return false;
        }
        if (i + need >= n) return false;
        for (size_t k = 1; k <= need; ++k)
            if ((((unsigned char)s[i + k]) & 0xC0) != 0x80) return false;
        i += need + 1;
    }
    return true;
}

static std::string token_display(const std::string &bytes) {
    if (valid_utf8(bytes)) return bytes;
    return "0x" + hex_of(bytes);
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

// Minimal RFC4180 reader: commas, quoted fields, "" escapes and embedded
// newlines. A '"' starts a quoted section only at the beginning of a field
// (else it is literal), matching Python's csv module for our corpora.
inline std::vector<std::vector<std::string>> read_csv_records(
    const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open CSV: " + path);
    std::string data((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    std::vector<std::vector<std::string>> records;
    std::vector<std::string> row;
    std::string field;
    bool in_quotes = false;
    size_t i = 0, n = data.size();
    auto end_field = [&]() {
        row.push_back(field);
        field.clear();
    };
    auto end_row = [&]() {
        end_field();
        // Skip the single trailing empty record produced by a final newline.
        records.push_back(row);
        row.clear();
    };
    while (i < n) {
        char c = data[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < n && data[i + 1] == '"') {
                    field.push_back('"');
                    i += 2;
                } else {
                    in_quotes = false;
                    ++i;
                }
            } else {
                field.push_back(c);
                ++i;
            }
        } else {
            if (c == '"' && field.empty()) {
                in_quotes = true;
                ++i;
            } else if (c == ',') {
                end_field();
                ++i;
            } else if (c == '\r' || c == '\n') {
                if (c == '\r' && i + 1 < n && data[i + 1] == '\n') i += 2;
                else ++i;
                end_row();
            } else {
                field.push_back(c);
                ++i;
            }
        }
    }
    if (in_quotes) throw std::runtime_error("unterminated quote in CSV: " + path);
    if (!field.empty() || !row.empty()) end_row();
    // Drop the artifact of a trailing newline: a final [""] record.
    if (!records.empty() && records.back().size() == 1 &&
        records.back()[0].empty())
        records.pop_back();
    return records;
}

// Values of `column`, skipping empty/missing ones (like `if text:` in the
// Python loaders).
inline std::vector<std::string> csv_column(
    const std::vector<std::vector<std::string>> &records,
    const std::string &column) {
    if (records.empty()) throw std::runtime_error("CSV has no header row");
    int idx = -1;
    for (size_t i = 0; i < records[0].size(); ++i)
        if (records[0][i] == column) idx = (int)i;
    if (idx < 0) {
        std::string avail;
        for (auto &h : records[0]) {
            if (!avail.empty()) avail += ",";
            avail += h;
        }
        throw std::runtime_error("column '" + column +
                                 "' not found (available: " + avail + ")");
    }
    std::vector<std::string> out;
    for (size_t r = 1; r < records.size(); ++r) {
        // Ragged rows: a missing field counts as empty (skipped), like
        // Python's DictReader mapping it to None.
        if (idx < (int)records[r].size() && !records[r][idx].empty())
            out.push_back(records[r][idx]);
    }
    return out;
}

static void csv_write_field(std::string &o, const std::string &f) {
    bool quote = f.find_first_of(",\"\r\n") != std::string::npos;
    if (!quote) {
        o += f;
        return;
    }
    o.push_back('"');
    for (char c : f) {
        if (c == '"') o += "\"\"";
        else o.push_back(c);
    }
    o.push_back('"');
}

// ---------------------------------------------------------------------------
// Pre-tokenization: split on specials, then `\s+\S+|\S+|\s+` per piece.
// ---------------------------------------------------------------------------

inline std::vector<std::string> pre_tokenize(
    const std::string &text, const std::vector<std::string> &specials) {
    std::vector<std::pair<bool, std::string>> segs; // (is_special, text)
    size_t pos = 0, n = text.size();
    while (pos < n) {
        size_t best_at = std::string::npos;
        size_t best_k = 0;
        for (size_t k = 0; k < specials.size(); ++k) {
            if (specials[k].empty()) continue;
            size_t at = text.find(specials[k], pos);
            if (at != std::string::npos &&
                (best_at == std::string::npos || at < best_at)) {
                best_at = at;
                best_k = k;
            }
        }
        if (best_at == std::string::npos) {
            segs.push_back({false, text.substr(pos)});
            break;
        }
        if (best_at > pos) segs.push_back({false, text.substr(pos, best_at - pos)});
        segs.push_back({true, specials[best_k]});
        pos = best_at + specials[best_k].size();
    }

    std::vector<std::string> out;
    for (auto &sg : segs) {        if (sg.first) {
            out.push_back(sg.second);
            continue;
        }
        const std::string &s = sg.second;
        size_t m = s.size(), i = 0;
        while (i < m) {
            if (is_space_byte((unsigned char)s[i])) {
                size_t j = i + 1;
                while (j < m && is_space_byte((unsigned char)s[j])) ++j;
                if (j < m && !is_space_byte((unsigned char)s[j])) {
                    size_t k = j + 1;
                    while (k < m && !is_space_byte((unsigned char)s[k])) ++k;
                    out.push_back(s.substr(i, k - i));
                    i = k;
                } else {
                    out.push_back(s.substr(i, j - i));
                    i = j;
                }
            } else {
                size_t k = i + 1;
                while (k < m && !is_space_byte((unsigned char)s[k])) ++k;
                out.push_back(s.substr(i, k - i));
                i = k;
            }
        }
    }
    return out;
}

// Distinct pre-tokens + counts in first-occurrence order (like
// dict(Counter(...)) in Python, which keeps insertion order).
inline std::pair<std::vector<std::string>, std::vector<Count>> count_words(
    const std::vector<std::string> &tokens) {
    std::vector<std::string> words;
    std::vector<Count> counts;
    std::unordered_map<std::string, size_t> idx;
    idx.reserve(tokens.size() * 2 + 1);
    for (auto &t : tokens) {
        auto f = idx.find(t);
        if (f == idx.end()) {
            idx.emplace(t, words.size());
            words.push_back(t);
            counts.push_back(1);
        } else {
            counts[f->second] += 1;
        }
    }
    return {words, counts};
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

struct TrainConfig {
    int steps = 100000;
    Count min_count = 5;
    long long max_vocab_size = 256 * 1024;
};

struct TrainProgress {
    int step = 0;
    int rule_count = 0;
    std::string new_token_display;
    Count frequency = 0;
    double compression = 0.0;
    bool stopped = false;
    std::string stop_reason;
};

static std::string word_key(const std::vector<Id> &toks) {
    std::string k;
    k.resize(toks.size() * 4);
    for (size_t i = 0; i < toks.size(); ++i) {
        uint32_t v = static_cast<uint32_t>(toks[i]);
        k[4 * i + 0] = (char)(v & 0xff);
        k[4 * i + 1] = (char)((v >> 8) & 0xff);
        k[4 * i + 2] = (char)((v >> 16) & 0xff);
        k[4 * i + 3] = (char)((v >> 24) & 0xff);
    }
    return k;
}

struct TrainWord {
    std::vector<Id> toks;
    Count count = 0;
    bool alive = true;
};

struct HeapEntry {
    Count freq;
    long long seq; // smaller (older) wins, like Counter.most_common ties
    PairKey pair;
};
struct HeapCmp {
    bool operator()(const HeapEntry &a, const HeapEntry &b) const {
        if (a.freq != b.freq) return a.freq < b.freq;
        return a.seq > b.seq;
    }
};

// Runs the merge loop. `words`/`counts` come from count_words().
// Progress callback is invoked every step (cheap); the CLI throttles output.
inline std::vector<std::pair<std::string, std::string>> train_bpe(
    const std::vector<std::string> &words, const std::vector<Count> &counts,
    const std::vector<std::string> &specials, const TrainConfig &cfg,
    const std::function<void(const TrainProgress &)> &progress = nullptr) {
    const int S = (int)specials.size();
    std::unordered_map<std::string, Id> special_id;
    for (int i = 0; i < S; ++i) special_id[specials[i]] = 256 + i;
    auto is_special = [S](Id id) { return id >= 256 && id < 256 + S; };

    std::vector<std::string> content;
    content.reserve(256 + S + cfg.steps);
    for (int b = 0; b < 256; ++b) content.push_back(std::string(1, (char)b));
    for (auto &s : specials) content.push_back(s);

    // Base split.
    std::vector<TrainWord> wv;
    wv.reserve(words.size());
    std::unordered_map<std::string, int> windex;
    windex.reserve(words.size() * 2 + 1);
    long long token_total = 0; // sum(len * count), for the compression ratio
    for (size_t i = 0; i < words.size(); ++i) {
        if (counts[i] <= 0) continue;
        std::vector<Id> toks;
        auto it = special_id.find(words[i]);
        if (it != special_id.end()) {
            toks.push_back(it->second);
        } else {
            toks.reserve(words[i].size());
            for (unsigned char b : words[i]) toks.push_back((Id)b);
            if (toks.empty()) continue;
        }
        std::string k = word_key(toks);
        auto f = windex.find(k);
        if (f != windex.end()) {
            wv[f->second].count += counts[i];
            token_total += (long long)toks.size() * counts[i];
        } else {
            int idx = (int)wv.size();
            windex.emplace(std::move(k), idx);
            TrainWord w;
            w.toks = std::move(toks);
            w.count = counts[i];
            token_total += (long long)w.toks.size() * w.count;
            wv.push_back(std::move(w));
        }
    }

    std::unordered_map<PairKey, Count> freq;
    std::unordered_map<PairKey, long long> seq;
    std::unordered_map<PairKey, std::unordered_set<int>> inv;
    freq.reserve(wv.size() * 2 + 1);
    inv.reserve(wv.size() * 2 + 1);
    long long next_seq = 0;
    for (int wi = 0; wi < (int)wv.size(); ++wi) {
        auto &toks = wv[wi].toks;
        Count c = wv[wi].count;
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            Id a = toks[i], b = toks[i + 1];
            if (is_special(a) || is_special(b)) continue;
            PairKey p = pack_pair(a, b);
            auto f = freq.find(p);
            if (f == freq.end()) {
                freq.emplace(p, c);
                seq.emplace(p, next_seq++);
            } else {
                f->second += c;
            }
            inv[p].insert(wi);
        }
    }
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;
    for (auto &kv : freq) heap.push({kv.second, seq[kv.first], kv.first});

    const long long cap = cfg.max_vocab_size - 256 - S;
    std::vector<std::pair<std::string, std::string>> rules;
    if (cap > 0) rules.reserve((size_t)std::min<long long>(cfg.steps, cap));
    std::vector<int> affected;
    affected.reserve(1024);

    auto merge_word = [&](const std::vector<Id> &toks, Id first, Id second,
                          Id rep) {
        std::vector<Id> out;
        out.reserve(toks.size());
        size_t i = 0, n = toks.size();
        while (i < n) {
            if (i + 1 < n && toks[i] == first && toks[i + 1] == second) {
                out.push_back(rep);
                i += 2;
            } else {
                out.push_back(toks[i]);
                i += 1;
            }
        }
        return out;
    };

    for (int step = 0; step < cfg.steps; ++step) {
        PairKey best_pair = 0;
        Count best_freq = 0;
        bool found = false;
        while (!heap.empty()) {
            HeapEntry top = heap.top();
            heap.pop();
            auto f = freq.find(top.pair);
            if (f == freq.end()) continue;
            if (f->second != top.freq) continue;
            auto s = seq.find(top.pair);
            if (s == seq.end() || s->second != top.seq) continue;
            best_pair = top.pair;
            best_freq = top.freq;
            found = true;
            break;
        }
        TrainProgress pr;
        pr.step = step;
        pr.rule_count = (int)rules.size();
        if (!found) {
            pr.stopped = true;
            pr.stop_reason = "no more pairs";
            if (progress) progress(pr);
            break;
        }
        if (best_freq < cfg.min_count) {
            pr.stopped = true;
            pr.stop_reason = "most common pair frequency " +
                             std::to_string(best_freq) + " < min_count " +
                             std::to_string(cfg.min_count);
            if (progress) progress(pr);
            break;
        }
        if ((long long)rules.size() >= cap) {
            pr.stopped = true;
            pr.stop_reason = "reached max_vocab_size=" +
                             std::to_string(cfg.max_vocab_size);
            if (progress) progress(pr);
            break;
        }

        Id first = pair_first(best_pair), second = pair_second(best_pair);
        Id new_id = (Id)(256 + S + (int)rules.size());
        std::string new_bytes = content[first] + content[second];
        if ((int)content.size() <= new_id) content.resize(new_id + 1);
        content[new_id] = new_bytes;

        auto inv_it = inv.find(best_pair);
        if (inv_it == inv.end()) continue;
        affected.assign(inv_it->second.begin(), inv_it->second.end());
        std::sort(affected.begin(), affected.end()); // deterministic order

        for (int wi : affected) {
            if (!wv[wi].alive) continue;
            TrainWord &w = wv[wi];
            Count c = w.count;
            std::vector<Id> old_toks = w.toks;
            if (old_toks.size() < 2) continue;

            for (size_t i = 0; i + 1 < old_toks.size(); ++i) {
                Id a = old_toks[i], b = old_toks[i + 1];
                if (is_special(a) || is_special(b)) continue;
                PairKey p = pack_pair(a, b);
                auto f = freq.find(p);
                if (f == freq.end()) continue;
                f->second -= c;
                auto iv = inv.find(p);
                if (iv != inv.end()) {
                    iv->second.erase(wi);
                    if (iv->second.empty()) inv.erase(iv);
                }
                if (f->second <= 0) {
                    freq.erase(f);
                    seq.erase(p);
                } else {
                    heap.push({f->second, seq[p], p});
                }
            }
            token_total -= (long long)old_toks.size() * c;

            std::vector<Id> new_toks = merge_word(old_toks, first, second, new_id);
            std::string old_k = word_key(old_toks);
            std::string new_k = word_key(new_toks);
            windex.erase(old_k);

            int target = wi;
            auto f2 = windex.find(new_k);
            if (f2 != windex.end() && f2->second != wi &&
                wv[f2->second].alive) {
                target = f2->second;
                wv[target].count += c;
                wv[wi].alive = false;
                wv[wi].toks.clear();
                wv[wi].count = 0;
            } else {
                w.toks = new_toks;
                windex.emplace(std::move(new_k), wi);
                token_total += (long long)new_toks.size() * c;
            }

            const std::vector<Id> &final_toks = wv[target].toks;
            for (size_t i = 0; i + 1 < final_toks.size(); ++i) {
                Id a = final_toks[i], b = final_toks[i + 1];
                if (is_special(a) || is_special(b)) continue;
                PairKey p = pack_pair(a, b);
                auto f = freq.find(p);
                if (f == freq.end()) {
                    freq.emplace(p, c);
                    seq.emplace(p, next_seq++);
                    heap.push({c, seq[p], p});
                } else {
                    f->second += c;
                    heap.push({f->second, seq[p], p});
                }
                inv[p].insert(target);
            }
        }

        rules.emplace_back(content[first], content[second]);
        if (progress) {
            pr.new_token_display = token_display(new_bytes);
            pr.frequency = best_freq;
            // compression vs the joined corpus byte length is filled by the
            // caller; here report vs current total only when available.
            pr.compression = 0.0;
            progress(pr);
        }
    }
    return rules;
}

// ---------------------------------------------------------------------------
// Minimal JSON (just enough for tokenizer.json, with correct string escapes)
// ---------------------------------------------------------------------------

struct Json {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;
    const Json &at(const std::string &k) const {
        for (auto &kv : obj)
            if (kv.first == k) return kv.second;
        throw std::runtime_error("missing JSON key: " + k);
    }
};

struct JsonParser {
    const char *p, *end;
    JsonParser(const std::string &s) : p(s.data()), end(s.data() + s.size()) {}
    [[noreturn]] void fail(const std::string &m) {
        throw std::runtime_error("JSON parse error: " + m);
    }
    void ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
            ++p;
    }
    static void utf8_out(std::string &o, unsigned cp) {
        if (cp < 0x80) o.push_back((char)cp);
        else if (cp < 0x800) {
            o.push_back((char)(0xC0 | (cp >> 6)));
            o.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            o.push_back((char)(0xE0 | (cp >> 12)));
            o.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            o.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            o.push_back((char)(0xF0 | (cp >> 18)));
            o.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            o.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            o.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }
    unsigned hex4() {
        if (end - p < 4) fail("bad \\u escape");
        unsigned v = 0;
        for (int i = 0; i < 4; ++i, ++p) {
            char c = *p;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
            else fail("bad \\u escape");
        }
        return v;
    }
    std::string string() {
        if (p >= end || *p != '"') fail("expected string");
        ++p;
        std::string o;
        while (p < end && *p != '"') {
            char c = *p;
            if (c == '\\') {
                ++p;
                if (p >= end) fail("bad escape");
                char e = *p++;
                switch (e) {
                    case '"': o.push_back('"'); break;
                    case '\\': o.push_back('\\'); break;
                    case '/': o.push_back('/'); break;
                    case 'b': o.push_back('\b'); break;
                    case 'f': o.push_back('\f'); break;
                    case 'n': o.push_back('\n'); break;
                    case 'r': o.push_back('\r'); break;
                    case 't': o.push_back('\t'); break;
                    case 'u': {
                        unsigned cp = hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                                p += 2;
                                unsigned lo = hex4();
                                if (lo >= 0xDC00 && lo <= 0xDFFF)
                                    cp = 0x10000 + ((cp - 0xD800) << 10) +
                                         (lo - 0xDC00);
                                else
                                    fail("bad surrogate pair");
                            } else {
                                fail("lone high surrogate");
                            }
                        }
                        utf8_out(o, cp);
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                o.push_back(c);
                ++p;
            }
        }
        if (p >= end) fail("unterminated string");
        ++p;
        return o;
    }
    Json value() {
        ws();
        if (p >= end) fail("unexpected end");
        Json j;
        char c = *p;
        if (c == '{') {
            ++p;
            j.type = Json::OBJ;
            ws();
            if (p < end && *p == '}') {
                ++p;
                return j;
            }
            while (true) {
                ws();
                std::string k = string();
                ws();
                if (p >= end || *p != ':') fail("expected ':'");
                ++p;
                Json v = value();
                j.obj.push_back({std::move(k), std::move(v)});
                ws();
                if (p >= end) fail("unterminated object");
                if (*p == ',') {
                    ++p;
                    continue;
                }
                if (*p == '}') {
                    ++p;
                    break;
                }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            ++p;
            j.type = Json::ARR;
            ws();
            if (p < end && *p == ']') {
                ++p;
                return j;
            }
            while (true) {
                j.arr.push_back(value());
                ws();
                if (p >= end) fail("unterminated array");
                if (*p == ',') {
                    ++p;
                    continue;
                }
                if (*p == ']') {
                    ++p;
                    break;
                }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            j.type = Json::STR;
            j.str = string();
        } else if ((c >= '0' && c <= '9') || c == '-') {
            const char *s = p;
            if (*p == '-') ++p;
            while (p < end && *p >= '0' && *p <= '9') ++p;
            if (p < end && *p == '.') {
                ++p;
                while (p < end && *p >= '0' && *p <= '9') ++p;
            }
            if (p < end && (*p == 'e' || *p == 'E')) {
                ++p;
                if (p < end && (*p == '+' || *p == '-')) ++p;
                while (p < end && *p >= '0' && *p <= '9') ++p;
            }
            j.type = Json::NUM;
            j.num = std::stod(std::string(s, p));
        } else if (end - p >= 4 && !strncmp(p, "true", 4)) {
            p += 4;
            j.type = Json::BOOL;
            j.b = true;
        } else if (end - p >= 5 && !strncmp(p, "false", 5)) {
            p += 5;
            j.type = Json::BOOL;
        } else if (end - p >= 4 && !strncmp(p, "null", 4)) {
            p += 4;
            j.type = Json::NUL;
        } else {
            fail("unexpected character");
        }
        return j;
    }
};

inline Json parse_json(const std::string &s) {
    JsonParser ps(s);
    Json j = ps.value();
    ps.ws();
    if (ps.p != ps.end) throw std::runtime_error("JSON parse error: trailing data");
    return j;
}

inline std::string json_escape(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 2);
    o.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b"; break;
            case '\f': o += "\\f"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o.push_back((char)c);
                }
        }
    }
    o.push_back('"');
    return o;
}

// ---------------------------------------------------------------------------
// Tokenizer model: TokenizerData <-> Vocab <-> Encoder
// ---------------------------------------------------------------------------

struct Token {
    bool is_str = false; // true: special token (text); false: raw bytes
    std::string data;
};

struct TokenizerData {
    std::vector<std::string> specials;
    std::vector<std::pair<std::string, std::string>> merges; // byte strings
    std::vector<Token> id_to_token;
};

inline Token token_from_json(const Json &o) {
    const std::string &t = o.at("type").str;
    if (t == "str") return Token{true, o.at("text").str};
    if (t == "bytes") return Token{false, bytes_of_hex(o.at("hex").str)};
    throw std::runtime_error("unknown token type: " + t);
}

inline TokenizerData load_tokenizer_json(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open tokenizer: " + path);
    std::string data((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    Json root = parse_json(data);
    if ((int)root.at("version").num != 1)
        throw std::runtime_error("unsupported tokenizer version");
    TokenizerData td;
    for (auto &s : root.at("special_tokens").arr) {
        if (s.type != Json::STR) throw std::runtime_error("bad special_token");
        td.specials.push_back(s.str);
    }
    for (auto &m : root.at("merges").arr) {
        if (m.type != Json::ARR || m.arr.size() != 2)
            throw std::runtime_error("bad merge entry");
        td.merges.push_back({token_from_json(m.arr[0]).data,
                             token_from_json(m.arr[1]).data});
    }
    for (auto &t : root.at("id_to_token").arr)
        td.id_to_token.push_back(token_from_json(t));
    return td;
}

// Vocabulary mirroring build_tokenizer_from_rules: 256 bytes + specials +
// merged tokens (first occurrence wins).
struct Vocab {
    std::vector<Token> id_to_token;
    std::unordered_map<std::string, OutId> bytes_to_id;
    std::unordered_map<std::string, OutId> special_to_id;
};

inline Vocab build_vocab(const std::vector<std::pair<std::string, std::string>> &merges,
                         const std::vector<std::string> &specials) {
    Vocab v;
    for (int i = 0; i < 256; ++i) {
        std::string b(1, (char)i);
        v.bytes_to_id[b] = i;
        v.id_to_token.push_back(Token{false, b});
    }
    for (auto &s : specials) {
        OutId id = (OutId)v.id_to_token.size();
        v.special_to_id[s] = id;
        v.id_to_token.push_back(Token{true, s});
    }
    for (auto &m : merges) {
        std::string nb = m.first + m.second;
        if (v.bytes_to_id.find(nb) == v.bytes_to_id.end()) {
            OutId id = (OutId)v.id_to_token.size();
            v.bytes_to_id[nb] = id;
            v.id_to_token.push_back(Token{false, nb});
        }
    }
    return v;
}

inline void save_tokenizer_json(const std::string &path,
                                const std::vector<std::string> &specials,
                                const std::vector<std::pair<std::string, std::string>> &merges,
                                const std::vector<Token> &id_to_token) {
    std::string o;
    o += "{\"version\":1,\"special_tokens\":[";
    for (size_t i = 0; i < specials.size(); ++i) {
        if (i) o += ",";
        o += json_escape(specials[i]);
    }
    o += "],\"merges\":[";
    for (size_t i = 0; i < merges.size(); ++i) {
        if (i) o += ",";
        o += "[{\"type\":\"bytes\",\"hex\":\"" + hex_of(merges[i].first) +
             "\"},{\"type\":\"bytes\",\"hex\":\"" + hex_of(merges[i].second) +
             "\"}]";
    }
    o += "],\"id_to_token\":[";
    for (size_t i = 0; i < id_to_token.size(); ++i) {
        if (i) o += ",";
        if (id_to_token[i].is_str)
            o += "{\"type\":\"str\",\"text\":" + json_escape(id_to_token[i].data) + "}";
        else
            o += "{\"type\":\"bytes\",\"hex\":\"" + hex_of(id_to_token[i].data) + "\"}";
    }
    o += "]}";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write: " + path);
    f << o;
}

inline void save_vocab_csv(const std::string &path,
                           const std::vector<Token> &id_to_token) {
    std::string o = "Token ID,Token\r\n";
    for (size_t i = 0; i < id_to_token.size(); ++i) {
        o += std::to_string(i);
        o += ",";
        std::string disp = id_to_token[i].is_str
                               ? id_to_token[i].data
                               : token_display(id_to_token[i].data);
        csv_write_field(o, disp);
        o += "\r\n";
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write: " + path);
    f << o;
}

// ---------------------------------------------------------------------------
// Encoder: replays merges in rank order. The per-row algorithm is exactly the
// Python `_merge_to_fixed_point` (repeatedly merge every occurrence of the
// lowest-rank pair present), but each "pass" runs over a linked list with a
// heap so it costs O(merges) instead of O(len) per pass.
// Read-only after construction -> safe to share across encoding threads.
// ---------------------------------------------------------------------------

class Encoder {
  public:
    std::vector<std::string> specials;
    std::unordered_map<std::string, int> special_idx;
    // Canonical byte-string table: 0..255 single bytes, then merged contents.
    std::vector<std::string> content;
    std::unordered_map<std::string, Id> content_id;
    std::unordered_map<PairKey, int> rank; // canonical pair -> merge index
    Vocab vocab;

    explicit Encoder(const TokenizerData &td) : specials(td.specials) {
        for (size_t i = 0; i < specials.size(); ++i)
            special_idx[specials[i]] = (int)i;
        content.reserve(256 + td.merges.size());
        for (int b = 0; b < 256; ++b) {
            std::string s(1, (char)b);
            content_id[s] = b;
            content.push_back(s);
        }
        auto canon = [&](const std::string &s) -> Id {
            auto f = content_id.find(s);
            if (f != content_id.end()) return f->second;
            Id id = (Id)content.size();
            content_id.emplace(s, id);
            content.push_back(s);
            return id;
        };
        for (size_t i = 0; i < td.merges.size(); ++i) {
            Id a = canon(td.merges[i].first);
            Id b = canon(td.merges[i].second);
            rank[pack_pair(a, b)] = (int)i; // same last-wins as Python dict
            canon(td.merges[i].first + td.merges[i].second);
        }
        vocab = build_vocab(td.merges, specials);
    }

    struct Item {
        bool special = false;
        int special_idx = -1;
        Id cid = -1;
        int prev = -1, next = -1;
        bool alive = false;
        int ver = 0;
    };
    struct Cand {
        int rank;
        int pos;
        int ver;
        Id first, second;
    };
    struct CandCmp {
        // lowest rank first; ties -> leftmost position (like Python's scan)
        bool operator()(const Cand &a, const Cand &b) const {
            if (a.rank != b.rank) return a.rank > b.rank;
            return a.pos > b.pos;
        }
    };

    std::vector<OutId> encode_text(const std::string &text) const {
        std::vector<std::string> pre = pre_tokenize(text, specials);
        std::vector<Item> items;
        items.reserve(pre.size() + 16);
        for (auto &t : pre) {
            auto f = special_idx.find(t);
            Item it;
            it.alive = true;
            if (f != special_idx.end()) {
                it.special = true;
                it.special_idx = f->second;
            } else {
                it.cid = -1; // expanded below into bytes
                for (unsigned char b : t) {
                    Item bi;
                    bi.alive = true;
                    bi.cid = (Id)b;
                    items.push_back(bi);
                }
                continue;
            }
            items.push_back(it);
        }
        int n = (int)items.size();
        for (int i = 0; i < n; ++i) {
            items[i].prev = i - 1;
            items[i].next = (i + 1 < n ? i + 1 : -1);
        }
        int head = n ? 0 : -1;

        std::priority_queue<Cand, std::vector<Cand>, CandCmp> heap;
        auto push_at = [&](int pos, auto &hq) {
            if (pos < 0 || !items[pos].alive || items[pos].special) return;
            int nx = items[pos].next;
            if (nx < 0 || !items[nx].alive || items[nx].special) return;
            auto f = rank.find(pack_pair(items[pos].cid, items[nx].cid));
            if (f == rank.end()) return;
            hq.push({f->second, pos, items[pos].ver, items[pos].cid,
                     items[nx].cid});
        };
        for (int i = 0; i < n; ++i) push_at(i, heap);

        // One "pass" per pop == one Python `_merge_to_fixed_point` iteration:
        // merge every current occurrence of the best pair, left to right.
        while (!heap.empty()) {
            Cand top = heap.top();
            heap.pop();
            int p = top.pos;
            if (p < 0 || p >= (int)items.size() || !items[p].alive ||
                items[p].special || items[p].ver != top.ver)
                continue;
            int nx = items[p].next;
            if (nx < 0 || !items[nx].alive || items[nx].special ||
                items[p].cid != top.first || items[nx].cid != top.second)
                continue;
            auto rf = rank.find(pack_pair(top.first, top.second));
            if (rf == rank.end() || rf->second != top.rank) continue;

            // Sweep: merge all occurrences of (first, second) left to right.
            int cur = head;
            while (cur != -1) {
                if (items[cur].alive && !items[cur].special &&
                    items[cur].cid == top.first) {
                    int nxt = items[cur].next;
                    if (nxt != -1 && items[nxt].alive && !items[nxt].special &&
                        items[nxt].cid == top.second) {
                        std::string nb =
                            content[top.first] + content[top.second];
                        // All merged contents were canonicalized at
                        // construction, so this lookup always hits.
                        auto cf = content_id.find(nb);
                        if (cf == content_id.end())
                            throw std::runtime_error(
                                "encoder: unknown merged content");
                        Id nid = cf->second;
                        Item m;
                        m.alive = true;
                        m.cid = nid;
                        int prv = items[cur].prev, after = items[nxt].next;
                        m.prev = prv;
                        m.next = after;
                        int mi = (int)items.size();
                        items.push_back(m);
                        if (prv != -1) {
                            items[prv].next = mi;
                            items[prv].ver++;
                        } else {
                            head = mi;
                        }
                        if (after != -1) {
                            // NOTE: do NOT bump after.ver: the pair starting
                            // at `after` is unchanged, so its heap entries
                            // stay valid (bumping would orphan them and lose
                            // merges).
                            items[after].prev = mi;
                        }
                        items[cur].alive = false;
                        items[nxt].alive = false;
                        push_at(prv, heap);
                        push_at(mi, heap);
                        cur = after; // continue AFTER the merged token
                        continue;
                    }
                }
                cur = items[cur].next;
            }
        }

        std::vector<OutId> ids;
        for (int cur = head; cur != -1; cur = items[cur].next) {
            if (!items[cur].alive) continue; // safety; list is clean
            if (items[cur].special) {
                ids.push_back(vocab.special_to_id.at(
                    specials[items[cur].special_idx]));
            } else {
                auto f = vocab.bytes_to_id.find(content[items[cur].cid]);
                if (f == vocab.bytes_to_id.end())
                    throw std::runtime_error(
                        "encoder: token not in vocabulary (was this "
                        "tokenizer trained on the same text?)");
                ids.push_back(f->second);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<OutId> &ids) const {
        std::string o;
        for (OutId id : ids) {
            if (id < 0 || id >= (OutId)vocab.id_to_token.size())
                throw std::runtime_error("decode: unknown token id");
            o += vocab.id_to_token[(size_t)id].data;
        }
        return o;
    }
};

} // namespace bpe
