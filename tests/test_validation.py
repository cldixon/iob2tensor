import pytest

from iob2labels.annotations import preprocessing, validate_spans


class TestValidateSpans:
    """Tests for upfront annotation validation with clear error messages."""

    def test_negative_start(self):
        with pytest.raises(ValueError, match="negative offset"):
            preprocessing(
                text="Hello world",
                spans=[{"start": -1, "end": 5, "label": "test"}],
            )

    def test_negative_end(self):
        with pytest.raises(ValueError, match="negative offset"):
            preprocessing(
                text="Hello world",
                spans=[{"start": 0, "end": -3, "label": "test"}],
            )

    def test_start_equals_end(self):
        with pytest.raises(ValueError, match="start.*>=.*end"):
            preprocessing(
                text="Hello world",
                spans=[{"start": 5, "end": 5, "label": "test"}],
            )

    def test_start_greater_than_end(self):
        with pytest.raises(ValueError, match="start.*>=.*end"):
            preprocessing(
                text="Hello world",
                spans=[{"start": 8, "end": 3, "label": "test"}],
            )

    def test_end_past_text_length(self):
        with pytest.raises(ValueError, match="extends past the text"):
            preprocessing(
                text="Hello",
                spans=[{"start": 0, "end": 10, "label": "test"}],
            )

    def test_overlapping_spans(self):
        with pytest.raises(ValueError, match="overlap"):
            preprocessing(
                text="Hello world test",
                spans=[
                    {"start": 0, "end": 8, "label": "a"},
                    {"start": 6, "end": 11, "label": "b"},
                ],
            )

    def test_overlapping_spans_reversed_order(self):
        """Overlap detection works regardless of span input order."""
        with pytest.raises(ValueError, match="overlap"):
            preprocessing(
                text="Hello world test",
                spans=[
                    {"start": 6, "end": 11, "label": "b"},
                    {"start": 0, "end": 8, "label": "a"},
                ],
            )

    def test_adjacent_spans_no_error(self):
        """Spans that touch (end == start of next) are valid, not overlapping."""
        result = preprocessing(
            text="Matt Damon Jason Bourne",
            spans=[
                {"start": 0, "end": 10, "label": "actor"},
                {"start": 10, "end": 23, "label": "character"},
            ],
        )
        assert len(result["spans"]) == 2

    def test_valid_annotation_passes(self):
        """A well-formed annotation passes validation without error."""
        result = preprocessing(
            text="Did Dame Judy Dench star in a film?",
            spans=[{"start": 4, "end": 19, "label": "actor"}],
        )
        assert result["text"] == "Did Dame Judy Dench star in a film?"
        assert len(result["spans"]) == 1

    def test_no_spans_passes(self):
        """An annotation with no spans passes validation."""
        result = preprocessing(text="No entities here.", spans=[])
        assert result["spans"] == []

    def test_error_message_includes_span_index_and_label(self):
        """Error messages reference the span index and label for debugging."""
        with pytest.raises(ValueError, match=r"Span 0.*'test'"):
            preprocessing(
                text="Hi",
                spans=[{"start": 0, "end": 100, "label": "test"}],
            )

    def test_overlap_message_includes_both_spans(self):
        """Overlap error references both offending spans."""
        with pytest.raises(ValueError, match=r"Spans \d+.*and \d+.*overlap"):
            preprocessing(
                text="Hello world test case",
                spans=[
                    {"start": 0, "end": 11, "label": "a"},
                    {"start": 5, "end": 15, "label": "b"},
                ],
            )
