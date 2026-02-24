# iob2labels

Convert [IOB2-format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) NER span annotations into integer label sequences for Transformer-based token classification tasks.

If you use annotation tools like [Prodigy](https://prodi.gy/docs), [Label Studio](https://labelstud.io/), or [Doccano](https://github.com/doccano/doccano) to annotate NER data, this library converts those character-offset span annotations into the label arrays you need for training.

## Installation

```bash
uv add iob2labels
```

Dependencies: `tokenizers` (HuggingFace Rust-backed tokenizer) and `pydantic`. No `torch` or `transformers` required.

## Quick Start

```python
from iob2labels import IOB2Encoder

encoder = IOB2Encoder(
    labels=["actor", "character", "plot"],
    tokenizer="bert-base-uncased",
)

labels = encoder(
    text="Did Dame Judy Dench star in a British film about Queen Elizabeth?",
    spans=[
        {"label": "actor", "start": 4, "end": 19},
        {"label": "plot", "start": 30, "end": 37},
        {"label": "character", "start": 49, "end": 64},
    ]
)
# >>> [-100, 0, 1, 2, 2, 2, 0, 0, 0, 5, 0, 0, 3, 4, 0, -100]
```

> Example pulled from the [MITMovie](https://groups.csail.mit.edu/sls/downloads/movie/) dataset.

The output is a `list[int]` aligned to the tokenizer's output. Convert to a tensor or array as needed:

```python
import torch
x = torch.tensor(labels)

# or with numpy
import numpy as np
x = np.array(labels)
```

## How It Works

The IOB2 format assigns each token one of three tag types:

- **O** (Outside) - not part of any entity
- **B-LABEL** (Beginning) - first token of an entity
- **I-LABEL** (Inside) - continuation of an entity

Each entity class generates 2 labels (B + I), plus the O class, so the total label count is always `(n * 2) + 1`:

```python
encoder.label_map
# >>> {'O': 0, 'B-ACTOR': 1, 'I-ACTOR': 2, 'B-CHARACTER': 3, 'I-CHARACTER': 4, 'B-PLOT': 5, 'I-PLOT': 6}
```

Special tokens (e.g., `[CLS]`, `[SEP]`) receive the ignore value `-100`, which PyTorch's `CrossEntropyLoss` skips by default.

## Tokenizer Input

The `tokenizer` argument accepts three forms:

```python
# 1. checkpoint name (downloads from HuggingFace Hub)
encoder = IOB2Encoder(labels=labels, tokenizer="bert-base-uncased")

# 2. standalone tokenizers.Tokenizer instance
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("bert-base-uncased")
encoder = IOB2Encoder(labels=labels, tokenizer=tok)

# 3. transformers PreTrainedTokenizerFast (unwrapped automatically)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
encoder = IOB2Encoder(labels=labels, tokenizer=tok)
```

## Batch Encoding

```python
annotations = [
    {"text": "Did Dame Judy Dench star?", "spans": [{"label": "actor", "start": 4, "end": 19}]},
    {"text": "Matt Damon was Jason Bourne.", "spans": [{"label": "actor", "start": 0, "end": 10}]},
]

results = encoder.batch(annotations)
# >>> [[-100, 0, 1, 2, 2, 2, 0, -100], [-100, 1, 2, 0, 0, 0, 0, -100]]
```

The batch path uses the Rust-backed `encode_batch()` for parallelized tokenization. Returns `list[list[int]]` with no padding; use HuggingFace's `DataCollatorForTokenClassification` or your own padding for training.

## Custom Field Names

If your annotation data uses non-standard field names, configure them at construction:

```python
# BioMed-NER dataset uses "entities" and "class" instead of "spans" and "label"
encoder = IOB2Encoder(
    labels=["organism", "chemicals"],
    tokenizer="bert-base-uncased",
    spans_field="entities",
    label_field="class",
)
```

## Built-in Conversion Check

By default, every encoding is verified by recovering the entity text from the produced labels and comparing it to the original annotation. This catches misalignment bugs early. Disable it for performance in production:

```python
encoder = IOB2Encoder(labels=labels, tokenizer=tok, conversion_check=False)
```

## Supported Tokenizers

Tested across three tokenizer families:

| Family | Checkpoints |
|---|---|
| WordPiece | `bert-base-cased`, `bert-base-uncased`, `bert-large-cased`, `bert-large-uncased`, `distilbert-base-cased`, `distilbert-base-uncased`, `google/electra-base-discriminator` |
| BPE | `roberta-base`, `roberta-large` |
| SentencePiece | `albert-base-v2`, `xlnet-base-cased`, `t5-small` |

Other HuggingFace-compatible tokenizers should work as well. The built-in conversion check will flag any issues.

## Tests

```bash
uv run pytest tests/ -v
```

The test suite includes unit tests for label map construction, entity range detection, and the conversion checker, plus a parametrized matrix of 12 tokenizer checkpoints across multiple annotation edge cases (entities at text boundaries, adjacent entities, punctuation, etc.).
