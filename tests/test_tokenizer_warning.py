import warnings
import pytest

from iob2labels import IOB2Encoder
from iob2labels.tokenizers import SUPPORTED_TOKENIZERS, resolve_tokenizer


class TestUntestedTokenizerWarning:
    """Tests for warning when using a tokenizer not in the supported list."""

    def test_supported_tokenizer_no_warning(self):
        """Supported checkpoints should not emit a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_tokenizer("bert-base-uncased")
        assert not any("not in the list" in str(w.message) for w in caught)

    def test_unsupported_tokenizer_warns(self):
        """An untested checkpoint should emit a UserWarning."""
        with pytest.warns(UserWarning, match="not in the list of checkpoints tested"):
            resolve_tokenizer("google/mobilebert-uncased")

    def test_warning_includes_checkpoint_name(self):
        """The warning message should include the checkpoint name used."""
        with pytest.warns(UserWarning, match="google/mobilebert-uncased"):
            resolve_tokenizer("google/mobilebert-uncased")

    def test_unsupported_via_encoder(self):
        """Warning surfaces when constructing IOB2Encoder with an untested checkpoint."""
        with pytest.warns(UserWarning, match="not in the list of checkpoints tested"):
            IOB2Encoder(labels=["person"], tokenizer="google/mobilebert-uncased")
