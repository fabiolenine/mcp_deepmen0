"""Property-style tests for the pure page-aware chunker."""

from __future__ import annotations

import re

import pytest

from mem0_mcp_selfhosted.chunking import Chunk, chunk_pages


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w{4,}", text.lower()))


class TestChunkPages:
    def test_small_pages_merge_into_one_chunk(self):
        pages = [(1, "Primeira página curta."), (2, "Segunda idem."), (3, "Terceira.")]
        chunks = chunk_pages(pages, chunk_chars=500)
        assert len(chunks) == 1
        assert chunks[0].page_start == 1 and chunks[0].page_end == 3
        assert "Primeira" in chunks[0].text and "Terceira" in chunks[0].text

    def test_no_chunk_exceeds_budget(self):
        big = " ".join(f"Sentença número {i} do parágrafo corrente." for i in range(200))
        pages = [(1, big), (2, "Página pequena final.")]
        chunks = chunk_pages(pages, chunk_chars=400, overlap=50)
        assert all(len(c.text) <= 400 for c in chunks)
        assert len(chunks) > 3

    def test_total_coverage_no_words_lost(self):
        paras = "\n\n".join(
            f"Parágrafo {i} fala sobre o tópico exclusivo palavrachave{i}." for i in range(60)
        )
        pages = [(1, paras), (2, "Encerramento com palavrafinal única.")]
        chunks = chunk_pages(pages, chunk_chars=350, overlap=40)
        combined = " ".join(c.text for c in chunks)
        assert _words(paras) | _words("Encerramento com palavrafinal única.") <= _words(combined)

    def test_page_ranges_are_accurate(self):
        big = "x" * 30 + ". " + ("Frase repetida para encher a página. " * 30)
        pages = [(3, "Página três pequena."), (4, big), (7, "Página sete depois de puladas.")]
        chunks = chunk_pages(pages, chunk_chars=300, overlap=0)
        for c in chunks:
            assert 3 <= c.page_start <= c.page_end <= 7
        split_pages = {c.page_start for c in chunks if c.page_start == c.page_end == 4}
        assert split_pages == {4}  # oversized page split stays attributed to page 4
        assert chunks[0].page_start == 3

    def test_intra_page_overlap_repeats_tail(self):
        sentences = [f"Fato numero {i} da lista continua." for i in range(40)]
        pages = [(1, " ".join(sentences))]
        chunks = chunk_pages(pages, chunk_chars=300, overlap=80)
        assert len(chunks) >= 2
        # the head of chunk N+1 repeats the tail of chunk N (straddle safety)
        tail = chunks[0].text[-40:]
        assert any(w in chunks[1].text for w in _words(tail))

    def test_indices_sequential_and_dataclass_shape(self):
        chunks = chunk_pages([(1, "Um conteúdo qualquer de página.")], chunk_chars=200)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert isinstance(chunks[0], Chunk)

    def test_empty_and_blank_pages_yield_nothing(self):
        assert chunk_pages([]) == []
        assert chunk_pages([(1, ""), (2, "   \n  ")]) == []

    def test_giant_unbroken_page_hard_splits(self):
        pages = [(1, "palavracontinua" * 500)]  # no paragraphs, no sentences
        chunks = chunk_pages(pages, chunk_chars=250, overlap=30)
        assert all(len(c.text) <= 250 for c in chunks)
        assert sum(len(c.text) for c in chunks) >= 500 * len("palavracontinua")

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            chunk_pages([(1, "x")], chunk_chars=0)
