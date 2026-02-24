import pytest
from tokenizers import Tokenizer

from iob2labels import IOB2Encoder

LABELS = ["actor", "character", "plot"]


class TestIOB2EncoderInit:
    """Test constructor behavior and properties."""

    def test_label_map_property(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        assert encoder.label_map == {
            "O": 0, "B-ACTOR": 1, "I-ACTOR": 2,
            "B-CHARACTER": 3, "I-CHARACTER": 4,
            "B-PLOT": 5, "I-PLOT": 6,
        }

    def test_label_map_is_copy(self):
        """Mutating the returned label_map should not affect the encoder."""
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        encoder.label_map["MUTATED"] = 99
        assert "MUTATED" not in encoder.label_map

    def test_string_tokenizer(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        assert isinstance(encoder.tokenizer, Tokenizer)

    def test_tokenizer_instance(self):
        tok = Tokenizer.from_pretrained("bert-base-uncased")
        encoder = IOB2Encoder(labels=LABELS, tokenizer=tok)
        assert isinstance(encoder.tokenizer, Tokenizer)

    def test_transformers_tokenizer(self):
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        encoder = IOB2Encoder(labels=LABELS, tokenizer=hf_tok)
        assert isinstance(encoder.tokenizer, Tokenizer)

    def test_invalid_tokenizer_type(self):
        with pytest.raises(TypeError):
            IOB2Encoder(labels=LABELS, tokenizer=12345)

    def test_invalid_checkpoint_name(self):
        with pytest.raises(ValueError):
            IOB2Encoder(labels=LABELS, tokenizer="this-model-does-not-exist-at-all")


class TestIOB2EncoderCall:
    """Test single-example encoding."""

    def test_basic_encoding(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        result = encoder(
            text="Did Dame Judy Dench star in a British film about Queen Elizabeth?",
            spans=[
                {"label": "actor", "start": 4, "end": 19},
                {"label": "plot", "start": 30, "end": 37},
                {"label": "character", "start": 49, "end": 64}
            ]
        )
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
        # <- verify specific known-good output for bert-base-uncased
        assert result == [-100, 0, 1, 2, 2, 2, 0, 0, 0, 5, 0, 0, 3, 4, 0, -100]

    def test_returns_list_int(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        result = encoder(text="Hello world", spans=[])
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_no_entities(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        result = encoder(text="No entities here.", spans=[])
        # all should be O (0) or ignore_token (-100)
        assert all(x in (0, -100) for x in result)

    def test_custom_field_names(self):
        encoder = IOB2Encoder(
            labels=["organism"],
            tokenizer="bert-base-uncased",
            start_field="begin",
            end_field="finish",
            label_field="class",
        )
        result = encoder(
            text="The cat sat on the mat",
            spans=[{"begin": 4, "finish": 7, "class": "organism"}]
        )
        assert isinstance(result, list)
        assert any(x not in (0, -100) for x in result)  # <- at least one entity label

    def test_truncation_skips_entities(self):
        encoder = IOB2Encoder(
            labels=LABELS,
            tokenizer="bert-base-uncased",
            max_length=10,
            conversion_check=False,  # <- checker would fail since entity count won't match
        )
        result = encoder(
            text="Did Dame Judy Dench star in a British film about Queen Elizabeth?",
            spans=[
                {"label": "actor", "start": 4, "end": 19},
                {"label": "character", "start": 49, "end": 64}  # <- beyond truncation
            ]
        )
        assert len(result) == 10


class TestIOB2EncoderBatch:
    """Test batch encoding."""

    def test_batch_basic(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        annotations = [
            {
                "text": "Did Dame Judy Dench star in a British film?",
                "spans": [{"label": "actor", "start": 4, "end": 19}]
            },
            {
                "text": "Matt Damon was Jason Bourne.",
                "spans": [
                    {"label": "actor", "start": 0, "end": 10},
                    {"label": "character", "start": 15, "end": 27}
                ]
            }
        ]
        results = encoder.batch(annotations)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_batch_variable_length(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        annotations = [
            {"text": "Short.", "spans": []},
            {"text": "This is a longer sentence for testing.", "spans": []}
        ]
        results = encoder.batch(annotations)
        assert len(results[0]) != len(results[1])  # <- different lengths, no padding

    def test_batch_on_error_skip(self):
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased", conversion_check=False)
        annotations = [
            {"text": "Hello world.", "spans": []},
            {"text": "Matt Damon is great.", "spans": [{"label": "actor", "start": 0, "end": 10}]},
        ]
        results = encoder.batch(annotations, on_error="skip")
        assert len(results) >= 1  # <- at least one should succeed
