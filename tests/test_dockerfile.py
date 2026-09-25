from pathlib import Path


def test_llama_install_only_packages_the_built_server():
    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    content = dockerfile.read_text()
    build_section = content.split("WORKDIR /src/llama.cpp", 1)[1].split("\nFROM --platform", 1)[0]

    assert "-DLLAMA_BUILD_EXAMPLES=OFF" in build_section
    assert "-DLLAMA_TOOLS_INSTALL=OFF" in build_section
    assert "install -D -m 0755 build/bin/libllama-server-impl.so /opt/llama/lib/libllama-server-impl.so" in build_section
