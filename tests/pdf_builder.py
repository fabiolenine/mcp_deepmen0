"""Minimal PDF builder for tests — pure stdlib, correct xref offsets.

Builds tiny valid PDFs (Helvetica/WinAnsiEncoding, so Latin-1 accents like
"ç"/"ã" survive pdftotext -enc UTF-8) without committing binary fixtures.
A page given as None carries no text operators — poppler extracts nothing,
which is exactly how a scanned page looks to pdftotext.
"""

from __future__ import annotations


def _escape(text: str) -> bytes:
    raw = text.encode("latin-1")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def build_pdf(pages: list[str | None]) -> bytes:
    """pages: one entry per page; a string becomes text lines, None = no text."""
    n = len(pages)
    page_nums = [3 + 2 * i for i in range(n)]
    content_nums = [4 + 2 * i for i in range(n)]
    font_num = 3 + 2 * n

    bodies: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            b"<< /Type /Pages /Kids ["
            + b" ".join(b"%d 0 R" % p for p in page_nums)
            + b"] /Count %d >>" % n
        ),
        font_num: (
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
            b" /Encoding /WinAnsiEncoding >>"
        ),
    }
    for i, text in enumerate(pages):
        bodies[page_nums[i]] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>"
            % (content_nums[i], font_num)
        )
        if text is None:
            stream = b""
        else:
            ops = [b"BT /F1 12 Tf 72 720 Td 14 TL"]
            for line in text.split("\n"):
                ops.append(b"(" + _escape(line) + b") Tj T*")
            ops.append(b"ET")
            stream = b" ".join(ops)
        bodies[content_nums[i]] = (
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(bodies):
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + bodies[num] + b"\nendobj\n"

    xref_pos = len(out)
    total = max(bodies) + 1
    out += b"xref\n0 %d\n" % total
    out += b"0000000000 65535 f \n"
    for num in range(1, total):
        out += b"%010d 00000 n \n" % offsets[num]
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (total, xref_pos)
    )
    return bytes(out)
