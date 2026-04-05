"""Property-based tests for ragstuffer — title extraction never crashes.

Uses hypothesis to generate arbitrary input and verify that
extract_text_with_title and related functions maintain their invariants.
"""

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from common import (
    ExtractedText,
    _extract_title_from_rst,
    _extract_title_from_text,
    extract_text_with_title,
)


@given(content=st.text(min_size=0, max_size=5000))
@settings(max_examples=200)
def test_extract_title_from_text_never_crashes(content: str) -> None:
    """_extract_title_from_text must not raise on arbitrary input."""
    result = _extract_title_from_text(content)
    assert isinstance(result, str)


@given(content=st.text(min_size=0, max_size=5000))
@settings(max_examples=200)
def test_extract_title_from_rst_never_crashes(content: str) -> None:
    """_extract_title_from_rst must not raise on arbitrary input."""
    result = _extract_title_from_rst(content)
    assert isinstance(result, str)


@given(content=st.text(min_size=1, max_size=2000))
@settings(max_examples=100)
def test_extract_text_with_title_txt_never_crashes(content: str) -> None:
    """extract_text_with_title for .txt files must not raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text(content)
        result = extract_text_with_title(test_file)
        assert isinstance(result, ExtractedText)
        assert isinstance(result.text, str)
        assert isinstance(result.title, str)


@given(content=st.text(min_size=1, max_size=2000))
@settings(max_examples=100)
def test_extract_text_with_title_md_never_crashes(content: str) -> None:
    """extract_text_with_title for .md files must not raise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.md"
        test_file.write_text(content)
        result = extract_text_with_title(test_file)
        assert isinstance(result, ExtractedText)
        assert isinstance(result.text, str)
        assert isinstance(result.title, str)


@given(
    title=st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_characters="\n\r#",
        ),
        min_size=1,
        max_size=100,
    ).filter(lambda t: t.strip() and not t.strip().startswith("#")),
)
@settings(max_examples=50)
def test_markdown_title_extraction_invariant(title: str) -> None:
    """If content starts with '# <title>', _extract_title_from_text returns that title."""
    content = f"# {title}\n\nSome body text."
    result = _extract_title_from_text(content)
    assert result == title.strip()
