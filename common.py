"""Shared constants and utilities for ragstuffer and ingest-remote.

Consolidates text extraction, chunking, file type constants, and Google Drive
constants that were previously duplicated between the two scripts.
"""

import logging
from pathlib import Path

log = logging.getLogger("ragstuffer.common")

# ── File type constants ───────────────────────────────────────────────────────

SUPPORTED_TEXT_EXTENSIONS = {".md", ".txt", ".adoc", ".rst"}
SUPPORTED_HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_BINARY_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx"}
ALL_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS | SUPPORTED_HTML_EXTENSIONS | SUPPORTED_BINARY_EXTENSIONS

# ── Google Drive constants ────────────────────────────────────────────────────

GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

EXPORT_MAP = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
}


# ── HTML text extraction ─────────────────────────────────────────────────────

def _make_html_parser():
    """Create an HTML parser that strips scripts/styles/nav and returns text."""
    from html.parser import HTMLParser

    class _HTMLTextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            self._skip = tag in ("script", "style", "nav", "header", "footer")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "header", "footer"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                t = data.strip()
                if t:
                    self.parts.append(t)

    return _HTMLTextExtractor()


# ── Text extraction ──────────────────────────────────────────────────────────


def extract_text(file_path: Path) -> str:
    """Extract text from a file based on its extension. Logs warnings on failure."""
    ext = file_path.suffix.lower()

    if ext in SUPPORTED_TEXT_EXTENSIONS:
        return file_path.read_text(errors="replace")

    if ext in SUPPORTED_HTML_EXTENSIONS:
        try:
            raw = file_path.read_text(errors="replace")
            parser = _make_html_parser()
            parser.feed(raw)
            return "\n".join(parser.parts)
        except Exception:
            log.warning("Failed to extract text from HTML: %s", file_path.name)
            return ""

    if ext == ".pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(str(file_path))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            log.warning("Failed to extract text from PDF: %s", file_path.name)
            return ""

    if ext == ".docx":
        try:
            import docx

            doc = docx.Document(str(file_path))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            log.warning("Failed to extract text from DOCX: %s", file_path.name)
            return ""

    if ext == ".pptx":
        try:
            from pptx import Presentation

            prs = Presentation(str(file_path))
            texts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        texts.append(shape.text_frame.text)
            return "\n\n".join(texts)
        except Exception:
            log.warning("Failed to extract text from PPTX: %s", file_path.name)
            return ""

    if ext == ".xlsx":
        try:
            import openpyxl

            wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
            texts = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    vals = [str(c) for c in row if c is not None]
                    if vals:
                        texts.append(" | ".join(vals))
            return "\n".join(texts)
        except Exception:
            log.warning("Failed to extract text from XLSX: %s", file_path.name)
            return ""

    return ""


# ── Chunking ─────────────────────────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 128) -> list[str]:
    """Split text into overlapping chunks using paragraph/sentence boundaries.

    Tries to split on paragraph breaks first, then sentences, then words,
    falling back to character-level splits. Adapted from the chunking strategy
    in validatedpatterns-sandbox/vector-embedder.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def extract_html_text(html: str) -> str:
    """Extract text from an HTML string, stripping scripts/styles/nav."""
    parser = _make_html_parser()
    parser.feed(html)
    return "\n\n".join(parser.parts)
