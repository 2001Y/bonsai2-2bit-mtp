import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class DockerfileTests(unittest.TestCase):
    def setUp(self):
        self.content = (ROOT / "Dockerfile").read_text()

    def test_llama_install_only_packages_the_built_server(self):
        build_section = self.content.split("WORKDIR /src/llama.cpp", 1)[1].split(
            "\nFROM --platform", 1
        )[0]

        self.assertIn("-DLLAMA_BUILD_EXAMPLES=OFF", build_section)
        self.assertIn("-DLLAMA_TOOLS_INSTALL=OFF", build_section)
        self.assertIn(
            "install -D -m 0755 build/bin/libllama-server-impl.so /opt/llama/lib/libllama-server-impl.so",
            build_section,
        )

    def test_image_starts_the_direct_http_entrypoint(self):
        self.assertIn(
            "COPY --chmod=0755 docker-entrypoint.sh /app/docker-entrypoint.sh", self.content
        )
        self.assertIn('ENTRYPOINT ["/app/docker-entrypoint.sh"]', self.content)
        self.assertNotIn("/app/worker.py", self.content)

    def test_image_declares_runpod_load_balancer_http_and_health_defaults(self):
        self.assertRegex(self.content, r"(?m)^EXPOSE 8080$")
        self.assertRegex(self.content, r"(?m)^\s+PORT=8080\s*\\$")
        self.assertRegex(self.content, r"(?m)^\s+PORT_HEALTH=8080\s*\\$")
        self.assertRegex(self.content, r"(?m)^\s+HEALTH_CHECK_PATH=/health\s*\\$")

    def test_final_image_does_not_install_queue_worker_dependencies(self):
        runtime_section = self.content.split("FROM nvidia/cuda:", 2)[-1]
        self.assertNotIn("uv sync", runtime_section)
        self.assertNotIn("python3", runtime_section.lower())


if __name__ == "__main__":
    unittest.main()
