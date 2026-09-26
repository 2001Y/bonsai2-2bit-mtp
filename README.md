# Bonsai 2 27B MTP — RunPod Serverless Load Balancer

This repository builds a direct-HTTP RunPod Serverless worker for the [BoldingBuilds Ternary Bonsai 2 27B Abliterated PQ2_0-MTP GGUF](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF). The container starts the pinned PrismML `llama-server` directly; it is **not** a RunPod queue/job-handler image.

This is an integration project, not an official RunPod, PrismML, Qwen, or BoldingBuilds release.

> **Safety:** This model's refusal behavior has been removed. It may answer requests that the original model would refuse. Do not expose it to end users without an independently designed and tested safety layer, authentication, abuse controls, and monitoring. See the [upstream model card](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF/blob/25bdc69496884b958428e722e2c3d15bf0e3857d/README.md#safety).

## What the image contains

- **Platform:** Linux `amd64` container on an NVIDIA CUDA GPU.
- **CUDA image:** NVIDIA CUDA 12.8.1, Ubuntu 24.04.
- **Runtime:** [`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp/tree/prism-b10687-5d80cff) at tag `prism-b10687-5d80cff`, verified to resolve to commit `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`.
- **MTP fix:** `0001-qwen35-mtp-hadamard-inverse.patch` from the pinned model revision, SHA-256 checked during image build.
- **Inference:** native `draft-mtp`, one llama-server slot, Flash Attention, 32K default context, and `--reasoning off`.
- **Model:** `Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf`, 7,657,489,696 bytes (about 7.66 GB), SHA-256 `7aa43b9a42f5bebc170d54f45657a7ccad841dd1f5b94434b107945740b02e86`.
- **Runtime dependencies:** no Python interpreter, RunPod queue SDK, or HTTP proxy; the native `llama-server` process serves the API.

The model card reports that the MTP variant decodes about 37% faster, but **this repository does not provide an independent benchmark**. The card reports testing on CUDA `sm_86` only. The Dockerfile compiles kernels for `sm_86` for the RTX 3090 target; other GPUs are not validated here.

## HTTP interface

The container listens on `0.0.0.0:8080` and serves llama.cpp's OpenAI-compatible API:

- Health: `GET /health`
- Model list: `GET /v1/models`
- Chat: `POST /v1/chat/completions`
- Served model ID: `ternary-bonsai-2-27b-abliterated-mtp`

The pinned llama.cpp revision also documents a Responses API route, but this repository targets Chat Completions for the initial Hermes integration. No live RunPod or Hermes request has been verified. Streaming and request-duration behavior through RunPod's load balancer remain untested.

RunPod Load Balancing forwards HTTP requests to the worker. **Choose endpoint type `Load Balancer`, not `Queue`**; a queue endpoint instead expects RunPod job routes and cannot directly proxy these OpenAI API paths.

## Deploy from this GitHub repository

RunPod's [GitHub integration](https://docs.runpod.io/serverless/workers/github-integration) can build a Dockerfile from a selected repository/branch and lets you select the endpoint type in the console. When creating the endpoint:

1. Select this repository, branch `main`, and the root `Dockerfile`.
2. Select endpoint type **Load Balancer**.
3. Choose the RTX 3090 / `sm_86` compatible target for this build. The default `CUDA_ARCHS=86` is not a claim of support for other GPU architectures.
4. Configure the container's HTTP port as `8080`; set `PORT=8080`, `PORT_HEALTH=8080`, and health-check path `/health` (the image supplies these defaults).
5. For a cost-controlled first deployment, use minimum workers `0` and maximum workers `1`; keep endpoint authentication enabled. Do not put a RunPod key in this repository.

A GitHub push only updates source. It does not create an endpoint or configure Hermes. RunPod performs the image build/deploy after you initiate that flow in its console. The Dockerfile downloads the pinned 7.66 GB model and compiles CUDA code **on RunPod's build service**; it does not download the model to this Mac. RunPod currently documents a 30-minute Docker build timeout for GitHub builds, and this full build has not been verified to fit that limit.

RunPod documents a 2-minute wait for an available worker, a 5.5-minute per-request processing limit, and a 30 MB request/response limit for Load Balancing. The cold start and typical generation time for this 7.66 GB model have not been measured; with minimum workers `0`, a first request may time out while the worker starts. Do not send a test request until you have reviewed the possible GPU charge and chosen to start a worker.

When deployed, the OpenAI-compatible base URL is:

```text
https://<ENDPOINT_ID>.api.runpod.ai/v1
```

For Hermes' standard custom endpoint, use API mode `chat_completions` and model ID `ternary-bonsai-2-27b-abliterated-mtp`. Keep the RunPod credential outside this repository and retrieve it through a supported local secret mechanism; do not copy it into this README or a committed config.

## Build inputs

The Dockerfile pins all large or mutable inputs:

| Input | Value |
| --- | --- |
| CUDA | `12.8.1` |
| llama.cpp tag / commit | `prism-b10687-5d80cff` / `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6` |
| Model repo revision | `25bdc69496884b958428e722e2c3d15bf0e3857d` |
| Model file SHA-256 | `7aa43b9a42f5bebc170d54f45657a7ccad841dd1f5b94434b107945740b02e86` |
| MTP patch SHA-256 | `ad6a3fac748a69d4d620906f534a01c36310e90a7f45adbe550ebc2f68718fba` |
| CUDA architectures | `86` (`sm_86`, RTX 3090) |

Do not change a model/runtime revision without verifying its associated hashes together. The final image includes the model weights; treat it as a model distribution and check the model license and registry terms before distributing the image.

## Static tests

These tests do not download the model, require a GPU, or build a Docker image:

```bash
python3 -m unittest discover -s tests -v
bash -n docker-entrypoint.sh
python3 -m compileall -q tests
```

A passing local test suite checks the entrypoint arguments and Dockerfile wiring; it does **not** prove that CUDA compilation, RunPod's remote build, GPU startup, or end-to-end inference succeeds.

## License and attribution

The worker source in this repository is licensed under Apache-2.0; see [`LICENSE`](LICENSE). The pinned model card declares Apache-2.0 for the model artifacts. The PrismML llama.cpp runtime is MIT-licensed. See [`NOTICE`](NOTICE) and the upstream project files for third-party attribution and terms.

- [Pinned model card and usage notes](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF/blob/25bdc69496884b958428e722e2c3d15bf0e3857d/README.md)
- [PrismML llama.cpp pinned source](https://github.com/PrismML-Eng/llama.cpp/tree/5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6)
- [Pinned llama-server API documentation](https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6/tools/server/README.md)
- [RunPod Load Balancing overview](https://docs.runpod.io/serverless/load-balancing/overview)
- [Build a RunPod Load Balancing worker](https://docs.runpod.io/serverless/load-balancing/build-a-worker)
- [RunPod GitHub integration](https://docs.runpod.io/serverless/workers/github-integration)
- [Docker build checks](https://docs.docker.com/build/checks/)
