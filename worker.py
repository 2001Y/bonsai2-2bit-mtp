import atexit
import os
import signal
import subprocess
import time
from typing import Any

import requests
import runpod

MODEL_ALIAS = "ternary-bonsai-2-27b-abliterated-mtp"
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))
LLAMA_URL = f"http://127.0.0.1:{LLAMA_PORT}"
STARTUP_TIMEOUT = int(os.getenv("LLAMA_STARTUP_TIMEOUT", "900"))
REQUEST_TIMEOUT = int(os.getenv("LLAMA_REQUEST_TIMEOUT", "900"))


def start_llama_server() -> subprocess.Popen:
    command = [
        "/opt/llama/bin/llama-server",
        "-m",
        os.environ.get("MODEL_PATH", "/models/model.gguf"),
        "--host",
        "127.0.0.1",
        "--port",
        str(LLAMA_PORT),
        "--alias",
        MODEL_ALIAS,
        "-ngl",
        "99",
        "-fa",
        "on",
        "-c",
        os.getenv("LLAMA_CTX_SIZE", "32768"),
        "--jinja",
        "--reasoning",
        "off",
        "--parallel",
        "1",
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        os.getenv("LLAMA_SPEC_DRAFT_N_MAX", "2"),
    ]
    print("Starting patched PrismML llama-server (MTP, thinking disabled)", flush=True)
    return subprocess.Popen(command, stdin=subprocess.DEVNULL)


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def wait_until_ready(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(
                f"llama-server exited during startup (exit={server.returncode})"
            )
        try:
            response = requests.get(f"{LLAMA_URL}/health", timeout=3)
            if response.ok:
                print("llama-server is ready", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"llama-server did not become ready within {STARTUP_TIMEOUT}s")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input") or {}
    if not isinstance(payload, dict):
        raise TypeError("input must be an object")

    request_body = dict(payload)
    if "messages" not in request_body:
        prompt = request_body.pop("prompt", None)
        if not isinstance(prompt, str):
            raise ValueError("input must contain OpenAI chat 'messages' or a string 'prompt'")
        request_body["messages"] = [{"role": "user", "content": prompt}]

    if request_body.get("stream"):
        raise ValueError("streaming is not supported by RunPod Serverless job responses")

    request_body["model"] = MODEL_ALIAS
    request_body["stream"] = False
    response = requests.post(
        f"{LLAMA_URL}/v1/chat/completions",
        json=request_body,
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(
            f"llama-server returned HTTP {response.status_code}: {response.text[:2000]}"
        )
    return response.json()


def main() -> None:
    server = start_llama_server()
    atexit.register(stop_server, server)

    def terminate(_signum: int, _frame: Any) -> None:
        stop_server(server)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    try:
        wait_until_ready(server)
        runpod.serverless.start({"handler": handler})
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
