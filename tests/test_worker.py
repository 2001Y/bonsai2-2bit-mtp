import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
ENTRYPOINT = ROOT / "docker-entrypoint.sh"


class LoadBalancerEntrypointTests(unittest.TestCase):
    def run_entrypoint(self, overrides=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            capture_path = temp / "argv.bin"
            server_path = temp / "fake-llama-server"
            server_path.write_text(
                "#!/bin/sh\nprintf '%s\\0' \"$@\" > \"$ARGS_CAPTURE\"\n",
                encoding="utf-8",
            )
            server_path.chmod(0o755)
            env = os.environ.copy()
            for name in (
                "PORT",
                "PORT_HEALTH",
                "MODEL_PATH",
                "LLAMA_CTX_SIZE",
                "LLAMA_SPEC_DRAFT_N_MAX",
            ):
                env.pop(name, None)
            env.update(
                {
                    "ARGS_CAPTURE": str(capture_path),
                    "LLAMA_SERVER_BIN": str(server_path),
                }
            )
            if overrides:
                env.update(overrides)

            result = subprocess.run(
                ["bash", str(ENTRYPOINT)],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            args = []
            if capture_path.exists():
                args = [
                    value.decode()
                    for value in capture_path.read_bytes().split(b"\0")
                    if value
                ]
            return result, args

    def test_serves_http_on_all_interfaces_and_uses_openai_model_alias(self):
        result, args = self.run_entrypoint()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index("-m") + 1], "/models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf")
        self.assertEqual(args[args.index("--host") + 1], "0.0.0.0")
        self.assertEqual(args[args.index("--port") + 1], "8080")
        self.assertEqual(args[args.index("--alias") + 1], "ternary-bonsai-2-27b-abliterated-mtp")

    def test_preserves_mtp_context_and_reasoning_settings(self):
        result, args = self.run_entrypoint()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index("-ngl") + 1], "99")
        self.assertEqual(args[args.index("-fa") + 1], "on")
        self.assertEqual(args[args.index("-c") + 1], "65536")
        self.assertIn("--jinja", args)
        self.assertEqual(args[args.index("--reasoning") + 1], "off")
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "2")

    def test_runpod_port_and_runtime_overrides_reach_llama_server(self):
        result, args = self.run_entrypoint(
            {
                "PORT": "9090",
                "MODEL_PATH": "/models/custom.gguf",
                "LLAMA_CTX_SIZE": "8192",
                "LLAMA_SPEC_DRAFT_N_MAX": "4",
            }
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index("-m") + 1], "/models/custom.gguf")
        self.assertEqual(args[args.index("--port") + 1], "9090")
        self.assertEqual(args[args.index("-c") + 1], "8192")
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "4")

    def test_rejects_a_health_port_that_differs_from_server_port(self):
        result, args = self.run_entrypoint({"PORT": "8080", "PORT_HEALTH": "8081"})

        self.assertEqual(result.returncode, 64)
        self.assertIn("PORT_HEALTH", result.stderr)
        self.assertEqual(args, [])


if __name__ == "__main__":
    unittest.main()
