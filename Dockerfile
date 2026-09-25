# syntax=docker/dockerfile:1.8
ARG CUDA_VERSION=12.8.1
ARG UV_VERSION=0.12.19
ARG MODEL_REPO=BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF
ARG MODEL_REVISION=25bdc69496884b958428e722e2c3d15bf0e3857d
ARG MODEL_FILENAME=Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf
ARG MODEL_SHA256=7aa43b9a42f5bebc170d54f45657a7ccad841dd1f5b94434b107945740b02e86
ARG PATCH_SHA256=ad6a3fac748a69d4d620906f534a01c36310e90a7f45adbe550ebc2f68718fba

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS llama-builder
ARG LLAMA_REF=prism-b10687-5d80cff
ARG LLAMA_COMMIT=5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6
ARG CUDA_ARCHS="75;80;86;89;90;100;120"
ARG BUILD_JOBS=4
ARG MODEL_REPO
ARG MODEL_REVISION
ARG PATCH_SHA256

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --filter=blob:none --branch "${LLAMA_REF}" \
      https://github.com/PrismML-Eng/llama.cpp.git llama.cpp \
    && test "$(git -C llama.cpp rev-parse HEAD)" = "${LLAMA_COMMIT}" \
    && curl -fL --retry 5 --retry-all-errors --retry-delay 3 \
      -o /tmp/0001-qwen35-mtp-hadamard-inverse.patch \
      "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/0001-qwen35-mtp-hadamard-inverse.patch" \
    && echo "${PATCH_SHA256}  /tmp/0001-qwen35-mtp-hadamard-inverse.patch" | sha256sum -c - \
    && git -C llama.cpp apply /tmp/0001-qwen35-mtp-hadamard-inverse.patch \
    && rm /tmp/0001-qwen35-mtp-hadamard-inverse.patch \
    && install -D -m 0644 /src/llama.cpp/LICENSE /licenses/llama.cpp-MIT.txt

WORKDIR /src/llama.cpp
RUN cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server --config Release -j"${BUILD_JOBS}" \
    && cmake --install build --prefix /opt/llama

FROM --platform=$BUILDPLATFORM ubuntu:24.04 AS model
ARG MODEL_REPO
ARG MODEL_REVISION
ARG MODEL_FILENAME
ARG MODEL_SHA256

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models \
    && curl -fL --retry 5 --retry-all-errors --retry-delay 5 \
      -o "/models/${MODEL_FILENAME}" \
      "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILENAME}" \
    && echo "${MODEL_SHA256}  /models/${MODEL_FILENAME}" | sha256sum -c -

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runpod-serverless
ARG MODEL_FILENAME
COPY --from=uv-bin /uv /uvx /bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 passwd python3 python3-venv \
    && groupadd --gid 10001 worker \
    && useradd --uid 10001 --gid worker --create-home --home-dir /home/worker worker \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY worker.py /app/worker.py
COPY LICENSE /licenses/Apache-2.0.txt
COPY NOTICE /licenses/NOTICE
COPY --from=llama-builder /opt/llama /opt/llama
COPY --from=llama-builder /licenses/llama.cpp-MIT.txt /licenses/llama.cpp-MIT.txt
COPY --from=model /models/${MODEL_FILENAME} /models/${MODEL_FILENAME}

ENV PATH="/opt/venv/bin:/opt/llama/bin:${PATH}" \
    HOME=/home/worker \
    LD_LIBRARY_PATH="/opt/llama/lib:/usr/local/cuda/lib64" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/${MODEL_FILENAME} \
    LLAMA_PORT=8080 \
    LLAMA_CTX_SIZE=32768 \
    LLAMA_SPEC_DRAFT_N_MAX=2 \
    LLAMA_STARTUP_TIMEOUT=900 \
    LLAMA_REQUEST_TIMEOUT=900

USER worker:worker
ENTRYPOINT ["/opt/venv/bin/python", "-u", "/app/worker.py"]
