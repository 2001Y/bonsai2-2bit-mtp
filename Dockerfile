# syntax=docker/dockerfile:1.8
ARG CUDA_VERSION=12.8.1
ARG MODEL_REPO=BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF
ARG MODEL_REVISION=25bdc69496884b958428e722e2c3d15bf0e3857d
ARG MODEL_FILENAME=Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf
ARG MODEL_SHA256=7aa43b9a42f5bebc170d54f45657a7ccad841dd1f5b94434b107945740b02e86
ARG PATCH_SHA256=ad6a3fac748a69d4d620906f534a01c36310e90a7f45adbe550ebc2f68718fba

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS llama-builder
ARG LLAMA_REF=prism-b10687-5d80cff
ARG LLAMA_COMMIT=5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6
# Build only for the RTX 3090 (sm_86) selected for this endpoint.
ARG CUDA_ARCHS="86"
ARG BUILD_JOBS=4
ARG MODEL_REPO
ARG MODEL_REVISION
ARG PATCH_SHA256

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY patches/runpod-health-initializing.patch /tmp/runpod-health-initializing.patch
RUN git clone --depth 1 --filter=blob:none --branch "${LLAMA_REF}" \
      https://github.com/PrismML-Eng/llama.cpp.git llama.cpp \
    && test "$(git -C llama.cpp rev-parse HEAD)" = "${LLAMA_COMMIT}" \
    && curl -fL --retry 5 --retry-all-errors --retry-delay 3 \
      -o /tmp/0001-qwen35-mtp-hadamard-inverse.patch \
      "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/0001-qwen35-mtp-hadamard-inverse.patch" \
    && echo "${PATCH_SHA256}  /tmp/0001-qwen35-mtp-hadamard-inverse.patch" | sha256sum -c - \
    && git -C llama.cpp apply /tmp/0001-qwen35-mtp-hadamard-inverse.patch \
    && rm /tmp/0001-qwen35-mtp-hadamard-inverse.patch \
    && git -C llama.cpp apply /tmp/runpod-health-initializing.patch \
    && rm /tmp/runpod-health-initializing.patch \
    && install -D -m 0644 /src/llama.cpp/LICENSE /licenses/llama.cpp-MIT.txt

WORKDIR /src/llama.cpp
RUN cmake -S . -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_TOOLS_INSTALL=OFF \
      -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target llama-server --config Release -j"${BUILD_JOBS}" \
    && cmake --install build --prefix /opt/llama \
    && install -D -m 0755 build/bin/libllama-server-impl.so /opt/llama/lib/libllama-server-impl.so

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

WORKDIR /app
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 passwd \
    && groupadd --gid 10001 worker \
    && useradd --uid 10001 --gid worker --create-home --home-dir /home/worker worker \
    && rm -rf /var/lib/apt/lists/*

COPY LICENSE /licenses/Apache-2.0.txt
COPY NOTICE /licenses/NOTICE
COPY --from=llama-builder /opt/llama /opt/llama
COPY --from=llama-builder /licenses/llama.cpp-MIT.txt /licenses/llama.cpp-MIT.txt
COPY --from=model /models/${MODEL_FILENAME} /models/${MODEL_FILENAME}
COPY --chmod=0755 docker-entrypoint.sh /app/docker-entrypoint.sh

ENV PATH="/opt/llama/bin:${PATH}" \
    HOME=/home/worker \
    LD_LIBRARY_PATH="/opt/llama/lib:/usr/local/cuda/lib64" \
    MODEL_PATH=/models/${MODEL_FILENAME} \
    PORT=8080 \
    PORT_HEALTH=8080 \
    HEALTH_CHECK_PATH=/health \
    LLAMA_CTX_SIZE=65536 \
    LLAMA_SPEC_DRAFT_N_MAX=2

EXPOSE 8080
USER worker:worker
ENTRYPOINT ["/app/docker-entrypoint.sh"]
