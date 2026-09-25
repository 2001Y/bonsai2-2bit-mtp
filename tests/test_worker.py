import importlib
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_import_does_not_start_llama_server(monkeypatch):
    sys.modules.pop("worker", None)

    def unexpected_start(*args, **kwargs):
        pytest.fail("importing worker.py must not spawn llama-server")

    monkeypatch.setitem(sys.modules, "runpod", SimpleNamespace(serverless=SimpleNamespace(start=lambda *_args, **_kwargs: None)))
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace())
    monkeypatch.setattr(subprocess, "Popen", unexpected_start)
    importlib.import_module("worker")
    sys.modules.pop("worker", None)


def test_chat_messages_are_forwarded_as_non_streaming_chat_completion(monkeypatch):
    worker = importlib.import_module("worker")
    response_body = {"choices": [{"message": {"content": "ok"}}]}
    response = SimpleNamespace(ok=True, json=lambda: response_body)
    observed = {}

    def fake_post(url, *, json, timeout):
        observed.update(url=url, body=json, timeout=timeout)
        return response

    monkeypatch.setattr(worker.requests, "post", fake_post)
    result = worker.handler(
        {
            "input": {
                "model": "caller-selected-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
            }
        }
    )

    assert result == response_body
    assert observed["url"].endswith("/v1/chat/completions")
    assert observed["body"]["model"] == worker.MODEL_ALIAS
    assert observed["body"]["stream"] is False
    assert observed["body"]["messages"] == [{"role": "user", "content": "hello"}]
    assert observed["body"]["max_tokens"] == 64


def test_prompt_shorthand_becomes_user_message(monkeypatch):
    worker = importlib.import_module("worker")
    captured = {}
    response = SimpleNamespace(ok=True, json=lambda: {"choices": []})

    def fake_post(_url, *, json, timeout):
        captured.update(body=json)
        return response

    monkeypatch.setattr(worker.requests, "post", fake_post)
    worker.handler({"input": {"prompt": "one prompt", "temperature": 0.25}})

    assert captured["body"]["messages"] == [{"role": "user", "content": "one prompt"}]
    assert captured["body"]["temperature"] == 0.25
    assert captured["body"]["stream"] is False


def test_streaming_is_rejected_for_queued_job_responses():
    worker = importlib.import_module("worker")

    with pytest.raises(ValueError, match="streaming is not supported"):
        worker.handler({"input": {"prompt": "hello", "stream": True}})


def test_non_object_input_raises_type_error():
    worker = importlib.import_module("worker")

    with pytest.raises(TypeError, match="input must be an object"):
        worker.handler({"input": ["not", "an", "object"]})


def test_worker_starts_server_with_mtp_and_thinking_disabled(monkeypatch):
    worker = importlib.import_module("worker")
    monkeypatch.setenv("MODEL_PATH", "/models/test.gguf")
    captured = {}
    fake_process = object()

    def fake_popen(command, *, stdin):
        captured.update(command=command, stdin=stdin)
        return fake_process

    monkeypatch.setattr(worker.subprocess, "Popen", fake_popen)
    result = worker.start_llama_server()

    assert result is fake_process
    command = captured["command"]
    assert command[command.index("-m") + 1] == "/models/test.gguf"
    assert "--spec-type" in command and command[command.index("--spec-type") + 1] == "draft-mtp"
    assert "--spec-draft-n-max" in command
    assert "--reasoning" in command and command[command.index("--reasoning") + 1] == "off"
    assert "--chat-template-kwargs" not in command
