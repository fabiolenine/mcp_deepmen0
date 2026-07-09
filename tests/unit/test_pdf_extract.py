"""Unit tests for poppler-based PDF extraction (PDFs built in-test, no binary fixtures)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from mem0_mcp_selfhosted.pdf_extract import (
    EncryptedPdf,
    ExtractionTimeout,
    FullyScannedPdf,
    PopplerMissing,
    extract_pages,
    pdf_info,
    rasterize_pages,
)
from tests.pdf_builder import build_pdf

poppler = pytest.mark.skipif(
    shutil.which("pdftotext") is None or shutil.which("pdfinfo") is None,
    reason="poppler-utils not installed",
)

PAGE1 = (
    "A informação do projeto Hermes fica registrada aqui.\n"
    "O gateway usa retry com jitter de 200ms nas chamadas."
)
PAGE2 = (
    "Segunda página: a decisão de arquitetura foi aprovada em 2026.\n"
    "A configuração padrão usa portas 8081 e 6333."
)


@pytest.fixture
def two_page_pdf(tmp_path):
    p = tmp_path / "digital.pdf"
    p.write_bytes(build_pdf([PAGE1, PAGE2]))
    return str(p)


@poppler
class TestPdfInfo:
    def test_reports_pages_and_not_encrypted(self, two_page_pdf):
        info = pdf_info(two_page_pdf)
        assert info["pages"] == 2
        assert info["encrypted"] is False

    def test_encrypted_detection_from_pdfinfo_output(self, two_page_pdf, monkeypatch):
        real_run = subprocess.run

        def fake_run(args, **kwargs):
            proc = real_run(args, **kwargs)
            if args[0] == "pdfinfo":
                proc = subprocess.CompletedProcess(
                    args, 0, stdout="Pages:          2\nEncrypted:      yes (print:no)\n",
                    stderr="",
                )
            return proc

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(EncryptedPdf):
            extract_pages(two_page_pdf)


@poppler
class TestExtractPages:
    def test_digital_pdf_pages_with_accents(self, two_page_pdf):
        pages = extract_pages(two_page_pdf)
        assert len(pages) == 2
        assert all(p.has_text for p in pages)
        assert pages[0].number == 1
        assert "informação" in pages[0].text  # WinAnsi ç/ã -> UTF-8 intactos
        assert "decisão" in pages[1].text

    def test_dehyphenation_rejoins_broken_words(self, tmp_path):
        p = tmp_path / "hyphen.pdf"
        p.write_bytes(build_pdf(["O sistema faz a informa-\nção fluir sem perdas no pipeline."]))
        pages = extract_pages(str(p))
        assert "informação" in pages[0].text
        assert "informa-" not in pages[0].text

    def test_fully_scanned_raises(self, tmp_path):
        p = tmp_path / "scanned.pdf"
        p.write_bytes(build_pdf([None, None]))
        with pytest.raises(FullyScannedPdf, match="v0.5b"):
            extract_pages(str(p))

    def test_mixed_pdf_classifies_per_page(self, tmp_path):
        p = tmp_path / "mixed.pdf"
        p.write_bytes(build_pdf([PAGE1, None, PAGE2]))
        pages = extract_pages(str(p))
        assert [pg.has_text for pg in pages] == [True, False, True]
        assert [pg.number for pg in pages] == [1, 2, 3]

    def test_timeout_maps_to_typed_error(self, two_page_pdf, monkeypatch):
        def boom(args, **kwargs):
            raise subprocess.TimeoutExpired(args, 1)

        monkeypatch.setattr(subprocess, "run", boom)
        with pytest.raises(ExtractionTimeout):
            extract_pages(two_page_pdf)

    def test_poppler_missing_is_typed(self, two_page_pdf, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(PopplerMissing):
            extract_pages(two_page_pdf)

    def test_errors_are_poison_compatible(self):
        # the worker classifies ValueError-derived exceptions as poison
        for exc in (EncryptedPdf, FullyScannedPdf, ExtractionTimeout, PopplerMissing):
            assert issubclass(exc, ValueError)


@poppler
class TestRasterize:
    def test_rasterizes_requested_pages_to_png(self, two_page_pdf):
        out = rasterize_pages(two_page_pdf, [1, 2], dpi=100)
        assert set(out) == {1, 2}
        assert all(v.startswith(b"\x89PNG\r\n\x1a\n") for v in out.values())

    def test_only_requested_page(self, two_page_pdf):
        out = rasterize_pages(two_page_pdf, [2], dpi=72)
        assert set(out) == {2}
