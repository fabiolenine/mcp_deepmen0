"""Unit tests for VLM vision transcription (v0.5b) — ollama client mocked."""

from __future__ import annotations

import pytest

import mem0_mcp_selfhosted.image_extract as ie
from mem0_mcp_selfhosted.image_extract import (
    VisionError,
    VisionUnavailable,
    transcribe_image,
    vision_enabled,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeClient:
    def __init__(self, content="texto transcrito", raises=None):
        self.content = content
        self.raises = raises
        self.chats = []
        self.unloaded = []

    def chat(self, **kwargs):
        self.chats.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return FakeResponse(self.content)

    def generate(self, **kwargs):
        if kwargs.get("keep_alive") == 0:
            self.unloaded.append(kwargs.get("model"))


@pytest.fixture
def vision_on(monkeypatch):
    monkeypatch.setenv("MEM0_ENABLE_VISION", "true")
    monkeypatch.setenv("MEM0_VLM_MODEL", "qwen3-vl:4b-instruct")
    monkeypatch.setenv("MEM0_LLM_MODEL", "extractor-model:latest")


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    # tolerante à assinatura: `_client` passou a receber o timeout do transporte
    monkeypatch.setattr(ie, "_client", lambda *a, **k: client)
    return client


class TestVisionEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("MEM0_ENABLE_VISION", raising=False)
        assert vision_enabled() is False

    def test_needs_both_flag_and_model(self, monkeypatch):
        monkeypatch.setenv("MEM0_ENABLE_VISION", "true")
        monkeypatch.delenv("MEM0_VLM_MODEL", raising=False)
        assert vision_enabled() is False
        monkeypatch.setenv("MEM0_VLM_MODEL", "qwen3-vl:4b-instruct")
        assert vision_enabled() is True


class TestTranscribe:
    def test_disabled_raises_unavailable(self, monkeypatch):
        monkeypatch.setenv("MEM0_ENABLE_VISION", "false")
        with pytest.raises(VisionUnavailable):
            transcribe_image(PNG)

    def test_transcribes_bytes(self, vision_on, fake_client):
        fake_client.content = "R$ 710,9 bilhões de produção setorial"
        out = transcribe_image(PNG)
        assert "710,9" in out
        call = fake_client.chats[0]
        assert call["model"] == "qwen3-vl:4b-instruct"
        assert call["messages"][0]["images"] == [PNG]
        assert call["options"]["temperature"] == 0

    def test_transcribes_path(self, vision_on, fake_client, tmp_path):
        p = tmp_path / "scan.png"
        p.write_bytes(PNG)
        transcribe_image(str(p))
        assert fake_client.chats[0]["messages"][0]["images"] == [PNG]

    def test_empty_transcription_is_poison(self, vision_on, fake_client):
        fake_client.content = "   "
        with pytest.raises(VisionError, match="empty"):
            transcribe_image(PNG)

    def test_request_failure_is_infra_not_poison(self, vision_on, monkeypatch):
        monkeypatch.setattr(ie, "_client", lambda *a, **k: FakeClient(raises=ConnectionError("ollama down")))
        # RuntimeError (retryable), NOT VisionError (poison) — infra failures retry
        with pytest.raises(RuntimeError):
            transcribe_image(PNG)
        with pytest.raises(Exception) as exc:
            transcribe_image(PNG)
        assert not isinstance(exc.value, ValueError)


class TestModelSwaps:
    def test_prepare_unloads_extractor(self, vision_on, fake_client):
        ie.prepare_vision()
        assert fake_client.unloaded == ["extractor-model:latest"]

    def test_release_unloads_vlm(self, vision_on, fake_client):
        ie.release_vision()
        assert fake_client.unloaded == ["qwen3-vl:4b-instruct"]

    def test_swaps_never_raise(self, vision_on, monkeypatch):
        class Boom:
            def generate(self, **k):
                raise RuntimeError("boom")
        monkeypatch.setattr(ie, "_client", lambda *a, **k: Boom())
        ie.prepare_vision()  # must not raise
        ie.release_vision()


class TestTimeoutReachesTheTransport:
    """`MEM0_VLM_TIMEOUT` era calculado e NUNCA aplicado.

    O worker de ingestão é serial e único: uma página travada no VLM prendia a
    fila inteira pelo default do httpx (sem limite de leitura em streaming).
    """

    def test_timeout_is_passed_to_the_client(self, monkeypatch):
        import mem0_mcp_selfhosted.image_extract as ie

        capturado = {}

        class _FakeClient:
            def __init__(self, host=None, timeout=None):
                capturado["timeout"] = timeout

            def chat(self, **kw):
                class _R:
                    class message:
                        content = "texto transcrito"
                return _R()

        monkeypatch.setenv("MEM0_ENABLE_VISION", "true")
        monkeypatch.setenv("MEM0_VLM_MODEL", "qwen3-vl:4b-instruct")
        monkeypatch.setenv("MEM0_VLM_TIMEOUT", "42")
        monkeypatch.setattr(ie, "_client",
                            lambda t=None: _FakeClient(timeout=t))
        ie.transcribe_image(b"\x89PNG\r\n\x1a\n")
        assert capturado["timeout"] == 42.0, (
            f"timeout não chegou ao transporte: {capturado}"
        )

    def test_explicit_timeout_s_wins_over_env(self, monkeypatch):
        import mem0_mcp_selfhosted.image_extract as ie

        capturado = {}
        monkeypatch.setenv("MEM0_ENABLE_VISION", "true")
        monkeypatch.setenv("MEM0_VLM_MODEL", "qwen3-vl:4b-instruct")
        monkeypatch.setenv("MEM0_VLM_TIMEOUT", "300")

        class _FakeClient:
            def __init__(self, host=None, timeout=None):
                capturado["timeout"] = timeout

            def chat(self, **kw):
                class _R:
                    class message:
                        content = "ok"
                return _R()

        monkeypatch.setattr(ie, "_client", lambda t=None: _FakeClient(timeout=t))
        ie.transcribe_image(b"\x89PNG\r\n\x1a\n", timeout_s=7)
        assert capturado["timeout"] == 7.0
