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
    monkeypatch.setenv("MEM0_LLM_MODEL", "llama3.1:8b")


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(ie, "_client", lambda: client)
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
        monkeypatch.setattr(ie, "_client", lambda: FakeClient(raises=ConnectionError("ollama down")))
        # RuntimeError (retryable), NOT VisionError (poison) — infra failures retry
        with pytest.raises(RuntimeError):
            transcribe_image(PNG)
        with pytest.raises(Exception) as exc:
            transcribe_image(PNG)
        assert not isinstance(exc.value, ValueError)


class TestModelSwaps:
    def test_prepare_unloads_extractor(self, vision_on, fake_client):
        ie.prepare_vision()
        assert fake_client.unloaded == ["llama3.1:8b"]

    def test_release_unloads_vlm(self, vision_on, fake_client):
        ie.release_vision()
        assert fake_client.unloaded == ["qwen3-vl:4b-instruct"]

    def test_swaps_never_raise(self, vision_on, monkeypatch):
        class Boom:
            def generate(self, **k):
                raise RuntimeError("boom")
        monkeypatch.setattr(ie, "_client", lambda: Boom())
        ie.prepare_vision()  # must not raise
        ie.release_vision()
