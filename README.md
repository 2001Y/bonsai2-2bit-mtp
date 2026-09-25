# Bonsai 2 27B MTP — RunPod Serverless Worker

A single-purpose **RunPod Serverless** worker for the [BoldingBuilds Ternary Bonsai 2 27B Abliterated PQ2_0-MTP GGUF](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF). The image builds a pinned CUDA-enabled PrismML `llama.cpp`, applies the model publisher's MTP fix, verifies the model checksum, and includes the model weights.

This is an integration project, not an official RunPod, PrismML, Qwen, or BoldingBuilds release.

> **Safety:** This model's refusal behavior has been removed. It may answer requests that the original model would refuse. Do not expose it to end users without an independently designed and tested safety layer, authentication, abuse controls, and monitoring. See the [upstream model card](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF/blob/25bdc69496884b958428e722e2c3d15bf0e3857d/README.md#safety).

## What it runs

- **Platform:** Linux `amd64` container on an NVIDIA CUDA GPU (RunPod Serverless target).
- **CUDA image:** NVIDIA CUDA 12.8.1, Ubuntu 24.04.
- **Runtime:** [`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp/tree/prism-b10687-5d80cff) at tag `prism-b10687-5d80cff`, verified to resolve to commit `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`.
- **MTP fix:** `0001-qwen35-mtp-hadamard-inverse.patch` from the pinned model revision, SHA-256 checked during image build.
- **Inference:** native `draft-mtp`, one llama-server slot, Flash Attention, 32K default context, and `--reasoning off`.
- **Python:** Python 3.12; dependencies are locked in `uv.lock`.
- **Model:** `Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf`, 7,657,489,696 bytes (about 7.66 GB), SHA-256 `7aa43b9a42f5bebc170d54f45657a7ccad841dd1f5b94434b107945740b02e86`.

The model card reports that the MTP variant decodes about 37% faster, but **this repository does not provide an independent benchmark**. The card reports testing on CUDA `sm_86` only; Metal and CPU paths are untested. The Dockerfile compiles kernels for a configurable set of NVIDIA compute capabilities, but compilation for an architecture is not evidence of model correctness or performance on that GPU.

### Platform support

| Platform | Status |
| --- | --- |
| RunPod Serverless, NVIDIA GPU, Linux `amd64` | Intended target; validate on the selected GPU before serving traffic. |
| Other NVIDIA GPUs | CUDA architectures are configurable at build time; not all GPU/model combinations have been tested. |
| Apple Silicon / Metal, CPU-only, AMD ROCm, Windows containers | Not implemented or verified by this project. |

This repository packages the CUDA worker only. It does not claim that MLX/Metal, CPU, ROCm, or other platforms support this patched MTP runtime.

## Model and image trade-offs

The GGUF is deliberately **baked into the image** so the worker does not need a Hugging Face token, network-volume setup, or a model download at runtime. The resulting image is large; registry transfer and cold-start time can therefore be substantial. Baking the weights is not a general cold-start optimization.

RunPod's current guidance recommends cached models for Hugging Face-hosted models when cold-start time and smaller images are the priority. The baked-image approach is a deliberate self-contained deployment option, not the RunPod default recommendation. Compare the [RunPod model-caching guide](https://docs.runpod.io/serverless/endpoints/model-caching) before choosing how to deploy.

## Build

Use a Buildx-enabled Docker installation. RunPod requires a `linux/amd64` image; its [Docker deployment guide](https://docs.runpod.io/serverless/workers/deploy) calls this out for Apple Silicon hosts. CUDA cross-builds under emulation may be slow or fail, so a native Linux `amd64` builder is preferred. The CUDA build does not need a GPU, but final inference does.

```bash
# Validate Dockerfile/build configuration without building the large image.
docker buildx build --check --platform linux/amd64 .

# Build for RunPod on a Linux/amd64 builder.
IMAGE=your-registry/your-namespace/bonsai2-2bit-mtp:0.1.0
docker buildx build \
  --platform linux/amd64 \
  --build-arg 'CUDA_ARCHS=86' \
  --tag "$IMAGE" \
  --push \
  .
```

`CUDA_ARCHS` defaults to `75;80;86;89;90;100;120` (Turing through Blackwell targets supported by the selected CUDA toolkit). Build only the architectures you need to reduce compile time and image size; shell-quote values containing semicolons. `BUILD_JOBS` defaults to `4`; lower it for emulated or memory-constrained builds and raise it only when the builder has headroom. The model publisher reports `sm_86` testing, so the example uses that architecture but does not imply other GPU support.

The build downloads the model and patch from the pinned Hugging Face revision and fails if either checksum differs. No model file, Hugging Face token, RunPod key, or registry credential belongs in the repository. The final image contains the model weights; treat it as a model distribution and secure the registry accordingly. Confirm the model license and registry terms before distributing an image.

## Configure a RunPod Serverless endpoint

1. Push the image to a registry RunPod can access and create a Serverless endpoint using that image. The image entrypoint starts the worker SDK; the internal llama-server listens only on container loopback and is not an inbound public port.
2. Select an NVIDIA GPU and configure the endpoint's worker execution timeout and concurrency for the model and expected prompt sizes. The defaults are not a guarantee that a particular GPU has enough VRAM; 32K context, weights, and runtime buffers all consume memory.
3. Keep the RunPod API key outside source control and client-side/browser code. Apply endpoint authentication and quotas appropriate to the data and users.
4. Use the asynchronous `/run` endpoint for cold starts or longer generation, then poll `/status/{job_id}`. `/runsync` is intended for jobs that complete inside its synchronous response window.

Example request (use an environment variable or secret manager; never replace the placeholder with a committed key):

```bash
curl --fail-with-body --request POST \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data '{"input":{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":128}}'

# Poll with the job id returned above.
curl --fail-with-body \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${JOB_ID}" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

The worker accepts `input.messages` in OpenAI Chat Completions form, or a shorthand `input.prompt` string. Other generation fields are forwarded to llama-server; the worker forces the configured model alias and `stream: false`. Streaming responses are rejected because RunPod Serverless returns completed job outputs.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_CTX_SIZE` | `32768` | Context size; reduce it if the selected GPU runs out of memory. |
| `LLAMA_SPEC_DRAFT_N_MAX` | `2` | Maximum MTP draft tokens. |
| `LLAMA_STARTUP_TIMEOUT` | `900` | Seconds to wait for llama-server health before failing worker startup. |
| `LLAMA_REQUEST_TIMEOUT` | `900` | Seconds allowed for an inference request to llama-server. |
| `LLAMA_PORT` | `8080` | Internal loopback port between the worker and llama-server; not a public endpoint. |
| `MODEL_PATH` | baked GGUF path | Override only when supplying a model at that path yourself. |

Build-time artifact inputs (`MODEL_REPO`, `MODEL_REVISION`, `MODEL_FILENAME`, `MODEL_SHA256`, `PATCH_SHA256`, `LLAMA_REF`, `LLAMA_COMMIT`, `CUDA_VERSION`, `CUDA_ARCHS`, and `BUILD_JOBS`) are declared in the Dockerfile. If you change a model revision or runtime ref, update and verify its corresponding commit/checksum values together.

## Development and tests

Python worker tests do not download the model or require a GPU:

```bash
uv sync --locked --group dev
uv run --locked --group dev pytest -q
uv run --locked python -m compileall -q worker.py tests
```

Dockerfile checks require Buildx 0.15 or later and a Docker daemon. A full image build downloads a 7.66 GB model and compiles the CUDA runtime; inference verification additionally requires an NVIDIA GPU. No throughput claim should be treated as verified until measured on the target GPU, context length, and request workload.

## License and attribution

The worker source in this repository is licensed under Apache-2.0; see [`LICENSE`](LICENSE). The pinned model card declares Apache-2.0 for the model artifacts. The PrismML llama.cpp runtime is MIT-licensed. See [`NOTICE`](NOTICE) and the upstream project files for third-party attribution and terms. The licenses of third-party Python packages remain with their respective publishers.

- [Pinned model card and usage notes](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF/blob/25bdc69496884b958428e722e2c3d15bf0e3857d/README.md)
- [PrismML llama.cpp pinned source](https://github.com/PrismML-Eng/llama.cpp/tree/5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6)
- [RunPod Serverless worker overview](https://docs.runpod.io/serverless/workers/overview)
- [RunPod Dockerfile guidance](https://docs.runpod.io/serverless/workers/create-dockerfile)
- [RunPod request guide](https://docs.runpod.io/serverless/endpoints/send-requests)
- [Docker build checks](https://docs.docker.com/build/checks/)
- [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/)
