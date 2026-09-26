import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class LoadBalancerReadinessTests(unittest.TestCase):
    def test_build_applies_health_patch_to_the_pinned_llama_server(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        patch_path = ROOT / "patches" / "runpod-health-initializing.patch"

        self.assertTrue(patch_path.is_file(), "the RunPod readiness patch must be checked in")
        self.assertIn(
            "COPY patches/runpod-health-initializing.patch /tmp/runpod-health-initializing.patch",
            dockerfile,
        )
        self.assertIn(
            "git -C llama.cpp apply /tmp/runpod-health-initializing.patch",
            dockerfile,
        )

    def test_initializing_status_is_health_only_and_ready_requests_still_work(self):
        patch_path = ROOT / "patches" / "runpod-health-initializing.patch"
        patch_text = patch_path.read_text() if patch_path.is_file() else ""

        self.assertIn('req.path == "/health"', patch_text)
        self.assertIn('req.path == "/v1/health"', patch_text)
        self.assertIn("res.status = 204;", patch_text)
        self.assertIn("res.status = 503;", patch_text)
        self.assertIn("if (!is_ready.load())", patch_text)

    def test_server_context_meets_hermes_minimum_by_default(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        entrypoint = (ROOT / "docker-entrypoint.sh").read_text()

        self.assertIn("LLAMA_CTX_SIZE=65536", dockerfile)
        self.assertIn(': "${LLAMA_CTX_SIZE:=65536}"', entrypoint)


if __name__ == "__main__":
    unittest.main()
