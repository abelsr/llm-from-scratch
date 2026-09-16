#!/bin/sh
# Build the standalone C++ BPE tools (no third-party dependencies).
#
#   sh scripts/build_cpp_tools.sh            # host or inside docker (phoson)
#
# Output: build/bin/train_tokenizer, build/bin/encode_corpus
set -e
ROOT="$(dirname "$0")/.."
OUT="$ROOT/build/bin"
mkdir -p "$OUT"
CXX="${CXX:-g++}"
FLAGS="-O3 -std=c++17 -DNDEBUG -Wall"
echo "+ $CXX train_tokenizer_cli.cpp -> $OUT/train_tokenizer"
$CXX $FLAGS -I"$ROOT/llm/cpp" "$ROOT/llm/cpp/train_tokenizer_cli.cpp" \
    -o "$OUT/train_tokenizer"
echo "+ $CXX encode_corpus_cli.cpp -> $OUT/encode_corpus"
$CXX $FLAGS -I"$ROOT/llm/cpp" "$ROOT/llm/cpp/encode_corpus_cli.cpp" \
    -o "$OUT/encode_corpus"
echo "Built $OUT/train_tokenizer $OUT/encode_corpus"
