FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /phoson

COPY pyproject.toml uv.lock .python-version  /phoson/

RUN uv sync

# Toolchain for the standalone C++ BPE tools (llm/cpp)
RUN apt-get update && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY data llm notebooks scripts main.py /phoson/

# Pre-build the standalone tools (dev compose mounts hide /phoson/build
# with host files, so rebuild there with sh scripts/build_cpp_tools.sh)
RUN sh scripts/build_cpp_tools.sh