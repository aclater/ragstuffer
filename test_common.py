"""Tests for common.py — extract_text, chunk_text, extract_html_text, constants.

NOTE: test_ragstuffer.py stubs langchain_text_splitters for module-level import.
We restore the real module here so chunk_text tests exercise the actual splitter.
"""

import sys
from unittest.mock import MagicMock

# Remove any MagicMock stub so the real langchain_text_splitters is used
if "langchain_text_splitters" in sys.modules and isinstance(
    sys.modules["langchain_text_splitters"], MagicMock
):
    del sys.modules["langchain_text_splitters"]

from pathlib import Path

from common import (
    ALL_EXTENSIONS,
    EXPORT_MAP,
    GDRIVE_SCOPES,
    SUPPORTED_BINARY_EXTENSIONS,
    SUPPORTED_HTML_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
    chunk_text,
    extract_html_text,
    extract_text,
)


# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_all_extensions_is_union(self):
        assert ALL_EXTENSIONS == (
            SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_HTML_EXTENSIONS | SUPPORTED_BINARY_EXTENSIONS
        )

    def test_gdrive_scopes_readonly(self):
        assert len(GDRIVE_SCOPES) == 1
        assert "readonly" in GDRIVE_SCOPES[0]

    def test_export_map_covers_google_docs(self):
        assert "application/vnd.google-apps.document" in EXPORT_MAP
        assert "application/vnd.google-apps.spreadsheet" in EXPORT_MAP
        assert "application/vnd.google-apps.presentation" in EXPORT_MAP


# ── extract_text ─────────────────────────────────────────────────────────────


class TestExtractText:
    def test_plain_text(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world")
        assert extract_text(f) == "Hello world"

    def test_markdown(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\n\nBody text")
        result = extract_text(f)
        assert "Title" in result
        assert "Body" in result

    def test_adoc(self, tmp_path):
        f = tmp_path / "test.adoc"
        f.write_text("= AsciiDoc Title\n\nParagraph here.")
        result = extract_text(f)
        assert "AsciiDoc Title" in result

    def test_rst(self, tmp_path):
        f = tmp_path / "test.rst"
        f.write_text("Title\n=====\n\nSome text.")
        result = extract_text(f)
        assert "Title" in result

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        assert extract_text(f) == ""

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        assert extract_text(f) == ""

    def test_html_strips_tags(self, tmp_path):
        f = tmp_path / "test.html"
        f.write_text("<html><body><p>Hello</p><script>evil()</script><p>World</p></body></html>")
        result = extract_text(f)
        assert "Hello" in result
        assert "World" in result
        assert "evil" not in result

    def test_html_strips_nav_header_footer(self, tmp_path):
        f = tmp_path / "test.htm"
        f.write_text(
            "<html><nav>Skip nav</nav><header>Skip header</header>"
            "<main>Keep this</main><footer>Skip footer</footer></html>"
        )
        result = extract_text(f)
        assert "Keep this" in result
        assert "Skip nav" not in result
        assert "Skip header" not in result
        assert "Skip footer" not in result

    def test_binary_without_library_returns_empty(self, tmp_path):
        """Binary formats return empty string if extraction library not available."""
        f = tmp_path / "test.xyz"
        f.write_bytes(b"\x00\x01")
        assert extract_text(f) == ""


# ── extract_html_text ────────────────────────────────────────────────────────


class TestExtractHtmlText:
    def test_basic_extraction(self):
        html = "<html><body><p>Hello</p><p>World</p></body></html>"
        result = extract_html_text(html)
        assert "Hello" in result
        assert "World" in result

    def test_strips_script_content(self):
        html = "<p>Before</p><script>alert('xss')</script><p>After</p>"
        result = extract_html_text(html)
        assert "Before" in result
        assert "After" in result
        assert "alert" not in result

    def test_strips_style_content(self):
        html = "<style>.foo { color: red; }</style><p>Visible</p>"
        result = extract_html_text(html)
        assert "Visible" in result
        assert "color" not in result

    def test_empty_html(self):
        assert extract_html_text("") == ""

    def test_plain_text_passthrough(self):
        result = extract_html_text("Just plain text")
        assert "Just plain text" in result


# ── chunk_text ───────────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world, this is a short text."
        chunks = chunk_text(text, chunk_size=1024, overlap=128)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = "Word " * 500  # ~2500 chars
        chunks = chunk_text(text, chunk_size=256, overlap=32)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        # Build text with clear sentence boundaries
        sentences = [f"Sentence number {i} is here." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        # With overlap, content from the end of one chunk should appear
        # at the start of the next
        assert len(chunks) > 2
        for i in range(len(chunks) - 1):
            # Last words of chunk i should appear somewhere in chunk i+1
            tail = chunks[i][-40:]
            # At least some overlap text should be in the next chunk
            overlap_found = any(word in chunks[i + 1] for word in tail.split() if len(word) > 3)
            assert overlap_found, f"No overlap between chunk {i} and {i+1}"

    def test_empty_text_returns_empty(self):
        assert chunk_text("", chunk_size=1024, overlap=128) == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n  ", chunk_size=1024, overlap=128) == []

    def test_respects_chunk_size(self):
        text = "Word " * 1000
        chunk_size = 512
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=64)
        for chunk in chunks:
            # Allow some tolerance — splitter may slightly exceed on word boundaries
            assert len(chunk) <= chunk_size * 1.1, f"Chunk too large: {len(chunk)} > {chunk_size * 1.1}"
