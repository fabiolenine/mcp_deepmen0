"""Vision transcription via a local Ollama VLM (v0.5b — images + scanned PDFs).

Turns an image (a standalone photo/scan, or a rasterized scanned PDF page)
into text so it can flow through the exact v0.5a chunk → fact-extraction
pipeline. Nothing leaves the host: pdftoppm rasterizes locally and the VLM
runs on the local Ollama GPU.

Model choice (measured on the , an 8GB GPU): ``qwen3-vl:4b-instruct``.
The bare ``qwen3-vl`` is a *thinking* model — it spends its whole token budget
reasoning and returns an EMPTY transcription; the ``-instruct`` variant
transcribes faithfully (accurate Portuguese, numbers and acronyms) at ~7s per
page when it has the GPU to itself.

VRAM reality: the 4B VLM (~2.9GB) and the resident llama3.1:8b extractor
(~5.5GB) do not fit together in 8GB. So a document is processed in two strict
phases — transcribe ALL pages with the VLM (extractor unloaded), then extract
facts from ALL text with llama3.1:8b (VLM unloaded) — two model swaps per job,
not two per page. The worker calls ``prepare_vision``/``release_vision`` to
force those swaps.
"""

from __future__ import annotations

import logging

from mem0_mcp_selfhosted.env import env
from mem0_mcp_selfhosted.pdf_extract import PdfExtractError

logger = logging.getLogger(__name__)

# Faithful transcription, no reasoning/description/summary — reused for both
# scanned pages and standalone images (an image's text IS its memorable content;
# a caption line covers the purely-pictorial case).
TRANSCRIPTION_PROMPT = (
    "Transcreva fielmente TODO o texto visível nesta imagem, na ordem de "
    "leitura, preservando números, siglas, datas e acentuação. Não descreva a "
    "imagem, não resuma e não comente — responda apenas com o texto transcrito. "
    "Se a imagem não contiver texto, descreva objetivamente o conteúdo visual "
    "em uma frase."
)


class VisionError(PdfExtractError):
    """Vision transcription failed for a bad-input reason (poison)."""


class VisionUnavailable(VisionError):
    """Vision requested but not configured/available."""


def vision_enabled() -> bool:
    return (env("MEM0_ENABLE_VISION", "false").lower() in ("true", "1", "yes")
            and bool(env("MEM0_VLM_MODEL")))


def _vlm_model() -> str:
    return env("MEM0_VLM_MODEL", "qwen3-vl:4b-instruct")


def _vlm_url() -> str:
    return env("MEM0_VLM_URL") or env("MEM0_LLM_URL") or env("MEM0_OLLAMA_URL") or "http://localhost:11434"


def _client():
    try:
        from ollama import Client
    except ImportError as e:
        raise VisionUnavailable(f"ollama client not installed: {e}") from None
    return Client(host=_vlm_url())


def _unload(model: str) -> None:
    """Best-effort eviction so the other model gets the full 8GB GPU."""
    try:
        _client().generate(model=model, prompt="", keep_alive=0)
    except Exception as e:
        logger.debug("Unload of %s failed (continuing): %s", model, e)


def prepare_vision() -> None:
    """Phase A start: evict the extraction model so the VLM owns the GPU."""
    extractor = env("MEM0_LLM_MODEL")
    if extractor:
        _unload(extractor)


def release_vision() -> None:
    """Phase A end: evict the VLM so the extractor reloads for phase B."""
    _unload(_vlm_model())


def transcribe_image(image: bytes | str, *, timeout_s: float | None = None) -> str:
    """Transcribe one image (bytes or a file path) to text via the VLM.

    Raises VisionUnavailable if vision is off/misconfigured, VisionError on a
    transcription failure — both poison (the worker sends them to dead-letter).
    """
    if not vision_enabled():
        raise VisionUnavailable("vision is disabled (set MEM0_ENABLE_VISION=true and MEM0_VLM_MODEL)")
    if isinstance(image, str):
        with open(image, "rb") as fh:
            image = fh.read()

    timeout = timeout_s if timeout_s is not None else float(env("MEM0_VLM_TIMEOUT", "300"))
    num_predict = int(env("MEM0_VLM_NUM_PREDICT", "2048"))
    # a full page at 150 DPI is ~4k vision tokens — the model's default 4096
    # context overflows (prompt alone exceeds it), so raise it explicitly to
    # hold the image + prompt + the transcription tokens
    num_ctx = int(env("MEM0_VLM_NUM_CTX", "8192"))
    client = _client()
    try:
        resp = client.chat(
            model=_vlm_model(),
            messages=[{"role": "user", "content": TRANSCRIPTION_PROMPT, "images": [image]}],
            options={"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx},
            keep_alive=env("MEM0_VLM_KEEP_ALIVE", "5m"),
        )
    except Exception as e:
        # network/timeout/connection = retryable infra; surface as-is so the
        # worker's classifier (not this module) decides. A ValueError-derived
        # VisionError would wrongly mark infra failures as poison.
        raise RuntimeError(f"VLM transcription request failed: {e}") from e

    text = (resp.message.content or "").strip()
    if not text:
        raise VisionError("VLM returned empty transcription (thinking model? use an -instruct variant)")
    return text
