# User Guide

## Overview

`iob2labels` converts character-offset NER span annotations (the format used by tools like [Prodigy](https://prodi.gy/docs), [Label Studio](https://labelstud.io/), and [Doccano](https://github.com/doccano/doccano)) into integer label sequences aligned to any HuggingFace-compatible tokenizer. At inference time, it converts model predictions back into span annotations.

The library depends only on `tokenizers` (HuggingFace Rust backend) and `pydantic`. No `torch` or `transformers` required.

## Installation

```bash
uv add iob2labels
```

Or with pip:

```bash
pip install iob2labels
```

## Setting Up the Encoder

The `IOB2Encoder` is the main interface. It requires two arguments: the entity class names and a tokenizer.

```python
from iob2labels import IOB2Encoder

encoder = IOB2Encoder(
    labels=["actor", "character", "plot"],
    tokenizer="bert-base-uncased",
)
```

### The `labels` Parameter

Pass a list of entity class names as strings. These are the NER categories in your annotation data. The encoder generates IOB2 tags from these labels:

- Each entity class produces 2 labels: `B-{LABEL}` (beginning) and `I-{LABEL}` (inside)
- Plus the `O` (outside) class
- Total label count is always `(n * 2) + 1`

```python
encoder.label_map
# {'O': 0, 'B-ACTOR': 1, 'I-ACTOR': 2, 'B-CHARACTER': 3, 'I-CHARACTER': 4, 'B-PLOT': 5, 'I-PLOT': 6}
```

### The `tokenizer` Parameter

The `tokenizer` argument accepts three forms:

=== "Checkpoint string"

    ```python
    encoder = IOB2Encoder(labels=labels, tokenizer="bert-base-uncased")
    ```

    Downloads the tokenizer from HuggingFace Hub. A `UserWarning` is emitted for checkpoints not in the [tested list](tokenizers.md).

=== "tokenizers.Tokenizer"

    ```python
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("bert-base-uncased")
    encoder = IOB2Encoder(labels=labels, tokenizer=tok)
    ```

=== "transformers PreTrainedTokenizerFast"

    ```python
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = IOB2Encoder(labels=labels, tokenizer=tok)
    ```

    The underlying `tokenizers.Tokenizer` is unwrapped automatically via the `.backend_tokenizer` attribute.

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ignore_token` | `int` | `-100` | Label value for special tokens (`[CLS]`, `[SEP]`, etc.). PyTorch's `CrossEntropyLoss` ignores this value by default. |
| `conversion_check` | `bool` | `True` | Verify encoding correctness via round-trip check after every encoding. Disable for production performance. |
| `max_length` | `int \| None` | `512` | Maximum token sequence length. Entities beyond the truncation boundary are skipped. Set to `None` to disable truncation. |

## Encoding Annotations

### Single Annotation

Call the encoder directly with `text` and `spans`:

```python
labels = encoder(
    text="Did Dame Judy Dench star in a British film about Queen Elizabeth?",
    spans=[
        {"label": "actor", "start": 4, "end": 19},
        {"label": "plot", "start": 30, "end": 37},
        {"label": "character", "start": 49, "end": 64},
    ]
)
# [-100, 0, 1, 2, 2, 2, 0, 0, 0, 5, 0, 0, 3, 4, 0, -100]
```

Each integer in the output corresponds to a token from the tokenizer:

- `-100` — special tokens (`[CLS]`, `[SEP]`), ignored during loss computation
- `0` — `O` (outside any entity)
- `1` — `B-ACTOR` (beginning of an actor entity)
- `2` — `I-ACTOR` (inside/continuation of an actor entity)
- And so on for each entity class

### Batch Encoding

For multiple annotations, use `batch()` which leverages the Rust-backed `encode_batch()` for parallelized tokenization:

```python
annotations = [
    {"text": "Did Dame Judy Dench star?", "spans": [{"label": "actor", "start": 4, "end": 19}]},
    {"text": "Matt Damon was Jason Bourne.", "spans": [{"label": "actor", "start": 0, "end": 10}]},
]

results = encoder.batch(annotations)
# [[-100, 0, 1, 2, 2, 2, 0, -100], [-100, 1, 2, 0, 0, 0, 0, -100]]
```

Results are returned without padding. Use HuggingFace's `DataCollatorForTokenClassification` or your own padding logic for training.

The `on_error` parameter controls error handling:

- `"raise"` (default) — raise on the first error
- `"skip"` — skip failed annotations, return results for successful ones

```python
results = encoder.batch(annotations, on_error="skip")
```

## Decoding Predictions

At inference time, convert model predictions (after `argmax`) back into character-offset span annotations.

### From Raw Text

Use `decode_text()` when you have the raw text but not the `Encoding` object:

```python
spans = encoder.decode_text(predicted_labels, text)
# [{"start": 4, "end": 19, "label": "actor"}, ...]
```

This tokenizes the text internally, then decodes.

### From a Pre-built Encoding

Use `decode()` when you already have the `tokenizers.Encoding` object (avoids re-tokenizing):

```python
encoding = encoder.tokenizer.encode(text)
spans = encoder.decode(predicted_labels, encoding, text)
```

Both methods return `list[Span]` — a list of typed dicts with `start`, `end`, and `label` fields.

!!! note "SentencePiece whitespace handling"
    Tokenizers like ALBERT, XLNet, T5, and XLM-RoBERTa absorb leading whitespace into tokens (e.g., `▁Queen` maps to chars `(48, 54)` instead of `(49, 54)`). The decoder corrects these offsets automatically using the original text, so the returned spans always have accurate character boundaries.

## Working with Custom Data Formats

Annotation tools use different field names. Configure the encoder to match your data format:

```python
# BioMed-NER dataset uses "entities" and "class" instead of "spans" and "label"
encoder = IOB2Encoder(
    labels=["organism", "chemicals"],
    tokenizer="bert-base-uncased",
    spans_field="entities",
    label_field="class",
)
```

Available field name overrides:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `text_field` | `"text"` | Key for the text string in batch annotation dicts |
| `spans_field` | `"spans"` | Key for the spans list in batch annotation dicts |
| `start_field` | `"start"` | Key for the start offset in span dicts |
| `end_field` | `"end"` | Key for the end offset in span dicts |
| `label_field` | `"label"` | Key for the entity label in span dicts |

## Annotation Validation

Input annotations are validated upfront with clear error messages:

- **Negative offsets** — `start` or `end` less than 0
- **Inverted spans** — `start >= end`
- **Out-of-bounds spans** — `end` exceeds text length
- **Overlapping spans** — IOB2 does not support overlapping entities

```python
encoder(text="Hello", spans=[{"label": "test", "start": 0, "end": 100}])
# ValueError: Span 0 ('test') extends past the text (end=100, text length=5).
# Ensure character offsets are within the text bounds.
```

## Conversion Checking

By default, every encoding is verified by recovering the entity text from the produced labels and comparing it to the original annotation. This catches tokenizer misalignment bugs early.

Disable it for production performance once you've verified correctness:

```python
encoder = IOB2Encoder(
    labels=labels,
    tokenizer=tok,
    conversion_check=False,
)
```

## Converting to Tensors

The encoder returns `list[int]`, which can be converted to any tensor format:

```python
import torch
x = torch.tensor(labels)

# or with numpy
import numpy as np
x = np.array(labels)
```

For batched training, the sequences are unpadded. Use HuggingFace's `DataCollatorForTokenClassification` to handle padding and label alignment:

```python
from transformers import DataCollatorForTokenClassification

collator = DataCollatorForTokenClassification(tokenizer=hf_tokenizer)
```
