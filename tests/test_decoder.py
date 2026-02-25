import pytest

from iob2labels import IOB2Encoder
from iob2labels.tokenizers import SUPPORTED_TOKENIZERS

from conftest import LABELS, STANDARD_ANNOTATION, MULTI_ANNOTATION, EDGE_CASE_ANNOTATIONS


## -- unit tests for decode / decode_text


class TestIOB2Decode:
    """Test decode and decode_text methods on IOB2Encoder."""

    def test_round_trip_basic(self):
        """Encode a standard annotation, decode the labels, verify recovered spans match."""
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        labels = encoder(**STANDARD_ANNOTATION)
        recovered = encoder.decode_text(labels, STANDARD_ANNOTATION["text"])
        assert recovered == STANDARD_ANNOTATION["spans"]

    def test_round_trip_multi_entity(self):
        """Encode a multi-entity annotation, decode, verify spans match."""
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        labels = encoder(**MULTI_ANNOTATION)
        recovered = encoder.decode_text(labels, MULTI_ANNOTATION["text"])
        assert recovered == MULTI_ANNOTATION["spans"]

    def test_no_entities(self):
        """All-O labels should decode to an empty span list."""
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        labels = encoder(text="No entities here.", spans=[])
        recovered = encoder.decode_text(labels, "No entities here.")
        assert recovered == []

    def test_decode_with_encoding_object(self):
        """decode() accepts an Encoding object directly (not just text)."""
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        text = STANDARD_ANNOTATION["text"]
        labels = encoder(**STANDARD_ANNOTATION)
        encoding = encoder.tokenizer.encode(text)
        recovered = encoder.decode(labels, encoding, text)
        assert recovered == STANDARD_ANNOTATION["spans"]

    def test_label_names_lowercase(self):
        """Recovered label names should be lowercase, matching the user's input convention."""
        encoder = IOB2Encoder(labels=["ACTOR", "CHARACTER"], tokenizer="bert-base-uncased")
        labels = encoder(
            text="Matt Damon starred as Jason Bourne",
            spans=[
                {"start": 0, "end": 10, "label": "ACTOR"},
                {"start": 22, "end": 34, "label": "CHARACTER"},
            ]
        )
        recovered = encoder.decode_text(labels, "Matt Damon starred as Jason Bourne")
        assert all(span["label"].islower() for span in recovered)
        assert recovered[0]["label"] == "actor"
        assert recovered[1]["label"] == "character"

    def test_ignore_tokens_skipped(self):
        """Ignore token positions (-100) should not affect decoded spans."""
        encoder = IOB2Encoder(labels=LABELS, tokenizer="bert-base-uncased")
        labels = encoder(**STANDARD_ANNOTATION)
        # <- verify ignore tokens are present in the label sequence
        assert labels[0] == -100  # [CLS]
        assert labels[-1] == -100  # [SEP]
        recovered = encoder.decode_text(labels, STANDARD_ANNOTATION["text"])
        assert recovered == STANDARD_ANNOTATION["spans"]

    def test_single_token_entity(self):
        """An entity that maps to exactly one token should round-trip correctly."""
        encoder = IOB2Encoder(labels=["plot"], tokenizer="bert-base-uncased")
        labels = encoder(text="A great film about love.", spans=[{"start": 19, "end": 23, "label": "plot"}])
        recovered = encoder.decode_text(labels, "A great film about love.")
        assert recovered == [{"start": 19, "end": 23, "label": "plot"}]


## -- parametrized round-trip tests across all supported tokenizers


@pytest.mark.parametrize("checkpoint", SUPPORTED_TOKENIZERS)
def test_round_trip_standard(checkpoint):
    """Round-trip: encode standard annotation → decode → assert spans match."""
    encoder = IOB2Encoder(labels=LABELS, tokenizer=checkpoint, conversion_check=True)
    labels = encoder(**STANDARD_ANNOTATION)
    recovered = encoder.decode_text(labels, STANDARD_ANNOTATION["text"])
    assert recovered == STANDARD_ANNOTATION["spans"], (
        f"Round-trip failed for {checkpoint}: recovered {recovered}"
    )


@pytest.mark.parametrize("checkpoint", SUPPORTED_TOKENIZERS)
def test_round_trip_multi(checkpoint):
    """Round-trip: encode multi-entity annotation → decode → assert spans match."""
    encoder = IOB2Encoder(labels=LABELS, tokenizer=checkpoint, conversion_check=True)
    labels = encoder(**MULTI_ANNOTATION)
    recovered = encoder.decode_text(labels, MULTI_ANNOTATION["text"])
    assert recovered == MULTI_ANNOTATION["spans"], (
        f"Round-trip failed for {checkpoint}: recovered {recovered}"
    )


@pytest.mark.parametrize("checkpoint", SUPPORTED_TOKENIZERS)
@pytest.mark.parametrize("case_name,annotation", list(EDGE_CASE_ANNOTATIONS.items()))
def test_round_trip_edge_cases(checkpoint, case_name, annotation):
    """Round-trip: encode each edge case → decode → assert spans match."""
    encoder = IOB2Encoder(labels=LABELS, tokenizer=checkpoint, conversion_check=True)
    labels = encoder(**annotation)
    recovered = encoder.decode_text(labels, annotation["text"])
    assert recovered == annotation["spans"], (
        f"Round-trip failed for edge case '{case_name}' with tokenizer '{checkpoint}': "
        f"recovered {recovered}, expected {annotation['spans']}"
    )
