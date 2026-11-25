"""Convert agent display text to speech-friendly plain text.

The agent may include a [SPOKEN]...[/SPOKEN] block in its reply with
a pre-written conversational summary.  When present, that block is
extracted and used as the spoken response.  The block is stripped from
the display response.

When no [SPOKEN] block exists, the display text is converted by:
  1. Removing markdown tables and replacing with natural-language rows.
  2. Stripping bold/italic/code markers.
  3. Collapsing bullet lists into flowing sentences.
  4. Removing headers, horizontal rules, and link syntax.
  5. Collapsing whitespace.

This module has no external dependencies beyond the standard library.
"""

from __future__ import annotations

import re
from typing import Tuple


# ------------------------------------------------------------------
# Spoken block extraction
# ------------------------------------------------------------------

_SPOKEN_RE = re.compile(
    r"\[SPOKEN\]\s*\n?(.*?)\n?\s*\[/SPOKEN\]",
    re.DOTALL | re.IGNORECASE,
)


def extract_spoken_block(text: str) -> Tuple[str, str | None]:
    """Return (display_text, spoken_text).

    If the text contains a [SPOKEN]...[/SPOKEN] block, the block is
    removed from display_text and its content returned as spoken_text.
    Otherwise spoken_text is None.
    """
    match = _SPOKEN_RE.search(text)
    if not match:
        return text, None

    spoken = match.group(1).strip()
    display = text[: match.start()] + text[match.end() :]
    display = display.strip()

    return display, spoken


# ------------------------------------------------------------------
# Markdown table → spoken prose
# ------------------------------------------------------------------

_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")
_SEPARATOR_RE = re.compile(r"^[\s|:\-]+$")


def _table_to_speech(table_lines: list[str]) -> str:
    """Convert a markdown table into spoken sentences."""
    rows: list[list[str]] = []
    headers: list[str] = []

    for line in table_lines:
        stripped = line.strip()
        if _SEPARATOR_RE.match(stripped):
            continue
        m = _TABLE_ROW_RE.match(stripped)
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if not headers:
            headers = cells
        else:
            rows.append(cells)

    if not rows:
        return ""

    parts: list[str] = []
    for i, row in enumerate(rows, 1):
        pairs = []
        for h, v in zip(headers, row):
            if v and v != "-" and v.lower() != "n/a":
                pairs.append(f"{h} is {v}")
        if pairs:
            if len(rows) > 1:
                parts.append(f"Option {i}: {', '.join(pairs)}.")
            else:
                parts.append(f"{', '.join(pairs)}.")

    return " ".join(parts)


# ------------------------------------------------------------------
# Full markdown → speech conversion
# ------------------------------------------------------------------

def markdown_to_speech(text: str) -> str:
    """Strip markdown formatting and produce speech-friendly text."""
    if not text:
        return ""

    lines = text.split("\n")
    result_lines: list[str] = []
    table_buffer: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()

        # Detect table rows
        is_table_line = bool(_TABLE_ROW_RE.match(stripped)) or bool(
            _SEPARATOR_RE.match(stripped) and "|" in stripped
        )

        if is_table_line:
            in_table = True
            table_buffer.append(stripped)
            continue

        if in_table:
            # Table just ended
            spoken = _table_to_speech(table_buffer)
            if spoken:
                result_lines.append(spoken)
            table_buffer = []
            in_table = False

        # Skip horizontal rules
        if re.match(r"^[-*_]{3,}$", stripped):
            continue

        # Strip markdown headers
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)

        # Strip bold and italic markers
        stripped = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", stripped)
        stripped = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", stripped)

        # Strip inline code
        stripped = re.sub(r"`([^`]+)`", r"\1", stripped)

        # Strip links: [text](url) → text
        stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)

        # Strip images
        stripped = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", stripped)

        # Convert bullet points to flowing text
        stripped = re.sub(r"^[\-*+]\s+", "", stripped)
        stripped = re.sub(r"^\d+\.\s+", "", stripped)

        if stripped:
            result_lines.append(stripped)

    # Flush remaining table
    if table_buffer:
        spoken = _table_to_speech(table_buffer)
        if spoken:
            result_lines.append(spoken)

    result = " ".join(result_lines)
    # Collapse multiple spaces
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def prepare_spoken_response(display_reply: str) -> Tuple[str, str]:
    """Return (display_text, spoken_text) from an agent reply.

    1. If the reply contains a [SPOKEN] block, extract it.
    2. Otherwise, convert the full display text to speech-friendly form.

    Returns the cleaned display text (without [SPOKEN] block) and the
    spoken text suitable for TTS.
    """
    display, spoken = extract_spoken_block(display_reply)

    if spoken:
        return display, spoken

    return display, markdown_to_speech(display)
