import pytest

from iob2labels import IOB2Encoder
from iob2labels.tokenizers import SUPPORTED_TOKENIZERS

from conftest import LABELS, STANDARD_ANNOTATION, MULTI_ANNOTATION, EDGE_CASE_ANNOTATIONS

## -- parametrized tests across all supported tokenizers and annotations


@pytest.mark.parametrize("checkpoint", SUPPORTED_TOKENIZERS)
def test_standard_annotation(checkpoint):
    """Every supported tokenizer should handle the standard annotation."""
    encoder = IOB2Encoder(
        labels=LABELS,
        tokenizer=checkpoint,
        conversion_check=True,
    )
    result = encoder(**STANDARD_ANNOTATION)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


@pytest.mark.parametrize("checkpoint", SUPPORTED_TOKENIZERS)
def test_multi_annotation(checkpoint):
    """Every supported tokenizer should handle multiple entity types."""
    encoder = IOB2Encoder(
        labels=LABELS,
        tokenizer=checkpoint,
        conversion_check=True,
    )
    result = encoder(**MULTI_ANNOTATION)
    assert isinstance(result, list)
    assert all(isinstance(x, int) for x in result)


@pytest.mark.parametrize("checkpoint", SUPPORTED_TOKENIZERS)
@pytest.mark.parametrize("case_name,annotation", list(EDGE_CASE_ANNOTATIONS.items()))
def test_edge_cases(checkpoint, case_name, annotation):
    """Every supported tokenizer should handle all edge case annotations."""
    encoder = IOB2Encoder(
        labels=LABELS,
        tokenizer=checkpoint,
        conversion_check=True,
    )
    try:
        result = encoder(**annotation)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
    except Exception as e:
        pytest.fail(
            f"Edge case '{case_name}' failed for tokenizer '{checkpoint}': {e}"
        )


@pytest.mark.parametrize("checkpoint", SUPPORTED_TOKENIZERS)
def test_batch_encoding(checkpoint):
    """Every supported tokenizer should handle batch encoding."""
    encoder = IOB2Encoder(labels=LABELS, tokenizer=checkpoint, conversion_check=True)
    annotations = [
        STANDARD_ANNOTATION,
        {"text": "No entities here.", "spans": []},
    ]
    results = encoder.batch(annotations)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)
