import warnings

from tokenizers import Tokenizer

## -- tokenizer checkpoints verified to work with iob2labels.
## -- organized by tokenizer family for clarity.
SUPPORTED_TOKENIZERS = [
    # WordPiece family
    "bert-base-cased",
    "bert-base-uncased",
    "bert-large-cased",
    "bert-large-uncased",
    "bert-base-multilingual-cased",
    "distilbert-base-cased",
    "distilbert-base-uncased",
    "google/electra-base-discriminator",
    # BPE family (byte-level)
    "roberta-base",
    "roberta-large",
    "distilroberta-base",
    "allenai/longformer-base-4096",
    # SentencePiece BPE family
    "FacebookAI/xlm-roberta-base",
    "almanach/camembert-base",
    # SentencePiece / Unigram family
    "albert-base-v2",
    "xlnet-base-cased",
    "t5-small",
    "google/flan-t5-base",
]

DEFAULT_MAX_LENGTH = 512


def resolve_tokenizer(tokenizer: str | Tokenizer) -> Tokenizer:
    """Normalize tokenizer input into a standalone tokenizers.Tokenizer instance.

    Accepts three forms:
      1. str               -- HuggingFace Hub checkpoint name, loaded via Tokenizer.from_pretrained()
      2. tokenizers.Tokenizer -- used directly
      3. transformers.PreTrainedTokenizerFast -- unwrapped via .backend_tokenizer attribute
         (detected via hasattr to avoid importing transformers)
    """
    if isinstance(tokenizer, str):
        if tokenizer not in SUPPORTED_TOKENIZERS:
            warnings.warn(
                f"Tokenizer '{tokenizer}' is not in the list of checkpoints tested with iob2labels. "
                f"It may work correctly, but results have not been verified. "
                f"Tested checkpoints: {SUPPORTED_TOKENIZERS}",
                UserWarning,
                stacklevel=3,  # <- surface warning at the caller's call site (through resolve_tokenizer -> IOB2Encoder)
            )
        try:
            return Tokenizer.from_pretrained(tokenizer)
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer from HuggingFace Hub for '{tokenizer}'. "
                f"Ensure the checkpoint name is correct and that the model has a tokenizer.json file. "
                f"Original error: {e}"
            ) from e

    if isinstance(tokenizer, Tokenizer):
        return tokenizer

    # <- detect transformers PreTrainedTokenizerFast without importing transformers
    if hasattr(tokenizer, "backend_tokenizer"):
        backend = tokenizer.backend_tokenizer
        if isinstance(backend, Tokenizer):
            return backend
        raise TypeError(
            f"tokenizer.backend_tokenizer is not a tokenizers.Tokenizer instance, "
            f"got {type(backend)}."
        )

    raise TypeError(
        f"Unsupported tokenizer type: {type(tokenizer)}. "
        f"Expected str (checkpoint name), tokenizers.Tokenizer, "
        f"or a transformers PreTrainedTokenizerFast with a .backend_tokenizer attribute."
    )
