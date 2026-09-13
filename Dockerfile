FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /phoson

COPY pyproject.toml uv.lock .python-version  /phoson/

RUN uv sync

COPY data llm notebooks scripts main.py /phoson/