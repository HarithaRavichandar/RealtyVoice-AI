"""Tests for the speech_text module — markdown-to-speech conversion."""

import pytest

from backend.orchestrator.services.speech_text import (
    extract_spoken_block,
    markdown_to_speech,
    prepare_spoken_response,
)


# ==================================================================
# extract_spoken_block
# ==================================================================

class TestExtractSpokenBlock:

    def test_extracts_spoken_block(self):
        text = (
            "Here are 3 apartments:\n\n"
            "| Name | Price |\n"
            "|------|-------|\n"
            "| A    | 50L   |\n\n"
            "[SPOKEN]I found three apartments for you.[/SPOKEN]"
        )
        display, spoken = extract_spoken_block(text)
        assert spoken == "I found three apartments for you."
        assert "[SPOKEN]" not in display
        assert "[/SPOKEN]" not in display
        assert "| Name |" in display

    def test_no_spoken_block_returns_none(self):
        text = "Hello, how can I help you?"
        display, spoken = extract_spoken_block(text)
        assert display == text
        assert spoken is None

    def test_spoken_block_case_insensitive(self):
        text = "Some text\n[spoken]Casual summary.[/spoken]"
        _, spoken = extract_spoken_block(text)
        assert spoken == "Casual summary."

    def test_multiline_spoken_block(self):
        text = (
            "Details above.\n"
            "[SPOKEN]\n"
            "I found two properties.\n"
            "The first is in Whitefield.\n"
            "[/SPOKEN]"
        )
        _, spoken = extract_spoken_block(text)
        assert "I found two properties." in spoken
        assert "The first is in Whitefield." in spoken


# ==================================================================
# markdown_to_speech
# ==================================================================

class TestMarkdownToSpeech:

    def test_strips_bold(self):
        assert "hello world" == markdown_to_speech("**hello world**")

    def test_strips_italic(self):
        assert "hello" == markdown_to_speech("*hello*")

    def test_strips_inline_code(self):
        assert "run this" == markdown_to_speech("`run this`")

    def test_strips_headers(self):
        result = markdown_to_speech("## Properties Found\nHere are your results.")
        assert result == "Properties Found Here are your results."

    def test_strips_links(self):
        result = markdown_to_speech("Visit [Google](https://google.com) for more.")
        assert result == "Visit Google for more."

    def test_strips_bullet_points(self):
        text = "- First item\n- Second item\n- Third item"
        result = markdown_to_speech(text)
        assert "First item" in result
        assert "- " not in result

    def test_strips_numbered_list(self):
        text = "1. First\n2. Second"
        result = markdown_to_speech(text)
        assert "1." not in result
        assert "First" in result

    def test_strips_horizontal_rule(self):
        result = markdown_to_speech("Above\n---\nBelow")
        assert "---" not in result
        assert "Above" in result
        assert "Below" in result

    def test_converts_simple_table(self):
        table = (
            "| Property | Price | Location |\n"
            "|----------|-------|----------|\n"
            "| Apt A    | 80L   | Whitefield |\n"
            "| Apt B    | 65L   | Sarjapur |\n"
        )
        result = markdown_to_speech(table)
        assert "Option 1" in result
        assert "Whitefield" in result
        assert "Option 2" in result
        assert "Sarjapur" in result
        assert "|" not in result

    def test_single_row_table_no_option_prefix(self):
        table = (
            "| Feature | Value |\n"
            "|---------|-------|\n"
            "| Bedrooms | 3 |\n"
        )
        result = markdown_to_speech(table)
        assert "Option" not in result
        assert "Bedrooms" in result

    def test_empty_string(self):
        assert markdown_to_speech("") == ""

    def test_plain_text_unchanged(self):
        text = "Hello, I can help you find an apartment."
        assert markdown_to_speech(text) == text

    def test_mixed_content(self):
        text = (
            "## Results\n\n"
            "I found **2 apartments**:\n\n"
            "| Name | Price |\n"
            "|------|-------|\n"
            "| Apt A | 50L |\n"
            "| Apt B | 70L |\n\n"
            "Would you like more details?"
        )
        result = markdown_to_speech(text)
        assert "##" not in result
        assert "**" not in result
        assert "|" not in result
        assert "Option 1" in result
        assert "Would you like more details?" in result


# ==================================================================
# prepare_spoken_response
# ==================================================================

class TestPrepareSpokenResponse:

    def test_uses_spoken_block_when_present(self):
        text = "Table here.\n[SPOKEN]Short summary.[/SPOKEN]"
        display, spoken = prepare_spoken_response(text)
        assert spoken == "Short summary."
        assert "[SPOKEN]" not in display

    def test_falls_back_to_markdown_conversion(self):
        text = "**Hello**, how can I help?"
        display, spoken = prepare_spoken_response(text)
        assert display == text
        assert "Hello, how can I help?" == spoken

    def test_plain_text_passes_through(self):
        text = "Welcome to RealtyVoice."
        display, spoken = prepare_spoken_response(text)
        assert display == text
        assert spoken == text
