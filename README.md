# iob2labels

Convert [IOB2-format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) NER span annotations into integer label sequences for Transformer-based token classification tasks — and convert them back.

If you use annotation tools like [Prodigy](https://prodi.gy/docs), [Label Studio](https://labelstud.io/), or [Doccano](https://github.com/doccano/doccano) to annotate NER data, this library converts those character-offset span annotations into the label arrays you need for training. At inference time, it converts model predictions back into span annotations.

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

## -- encoding: spans → labels (for training)
labels = encoder(
    text="Did Dame Judy Dench star in a British film about Queen Elizabeth?",
    spans=[
        {"label": "actor", "start": 4, "end": 19},
        {"label": "plot", "start": 30, "end": 37},
        {"label": "character", "start": 49, "end": 64},
    ]
)
# >>> [-100, 0, 1, 2, 2, 2, 0, 0, 0, 5, 0, 0, 3, 4, 0, -100]

## -- decoding: labels → spans (for inference)
spans = encoder.decode_text(labels, "Did Dame Judy Dench star in a British film about Queen Elizabeth?")
# >>> [{"start": 4, "end": 19, "label": "actor"}, {"start": 30, "end": 37, "label": "plot"}, {"start": 49, "end": 64, "label": "character"}]
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

## Decoding

At inference time, convert model predictions back into character-offset span annotations:

```python
## -- decode_text: pass the raw text (tokenizes internally)
spans = encoder.decode_text(predicted_labels, text)

## -- decode: pass a pre-built Encoding object (avoids re-tokenizing)
encoding = encoder.tokenizer.encode(text)
spans = encoder.decode(predicted_labels, encoding, text)
```

Both return `list[Span]` — a list of typed dicts with `start`, `end`, and `label` fields.

The decoder handles SentencePiece-style tokenizers (ALBERT, XLNet, T5, XLM-RoBERTa, CamemBERT) that absorb leading whitespace into tokens (e.g., `▁Queen` maps to chars `(48, 54)` instead of `(49, 54)`). Character offsets are corrected automatically using the original text.

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

When using a checkpoint string not in the [tested list](#supported-tokenizers), a `UserWarning` is emitted. The tokenizer will still load and may work correctly, but round-trip correctness has not been verified.

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

## Annotation Validation

Input annotations are validated upfront with clear error messages for common mistakes:

- Negative character offsets
- Inverted spans (`start >= end`)
- Spans extending past the text length
- Overlapping spans (IOB2 does not support overlapping entities)

```python
encoder(text="Hello", spans=[{"label": "test", "start": 0, "end": 100}])
# ValueError: Span 0 ('test') extends past the text (end=100, text length=5).
# Ensure character offsets are within the text bounds.
```

## Built-in Conversion Check

By default, every encoding is verified by recovering the entity text from the produced labels and comparing it to the original annotation. This catches misalignment bugs early. Disable it for performance in production:

```python
encoder = IOB2Encoder(labels=labels, tokenizer=tok, conversion_check=False)
```

## Supported Tokenizers

Tested across four tokenizer families:

| Family | Checkpoints |
|---|---|
| WordPiece | `bert-base-cased`, `bert-base-uncased`, `bert-large-cased`, `bert-large-uncased`, `bert-base-multilingual-cased`, `distilbert-base-cased`, `distilbert-base-uncased`, `google/electra-base-discriminator` |
| BPE (byte-level) | `roberta-base`, `roberta-large`, `distilroberta-base`, `allenai/longformer-base-4096` |
| SentencePiece BPE | `FacebookAI/xlm-roberta-base`, `almanach/camembert-base` |
| SentencePiece Unigram | `albert-base-v2`, `xlnet-base-cased`, `t5-small`, `google/flan-t5-base` |

Other HuggingFace-compatible tokenizers with a `tokenizer.json` file on the Hub should work as well. A `UserWarning` is emitted for untested checkpoints, and the built-in conversion check will flag any alignment issues.

## Tests

```bash
uv run pytest tests/ -v
```

313 tests across 10 test files, including:

- **Encoding matrix**: every supported tokenizer × standard, multi-entity, and edge case annotations (entity at start/end of text, adjacent entities, punctuation)
- **Decoding round-trips**: encode → decode → assert recovered spans exactly match originals, across all 18 tokenizers. This is the strongest correctness guarantee — if the round-trip holds, both the encoder and decoder are correct for that tokenizer.
- **Annotation validation**: negative offsets, inverted spans, overlapping entities, spans past text bounds
- **Tokenizer warning**: untested checkpoint detection
- **SentencePiece whitespace handling**: tokenizers like `google/flan-t5-base` absorb leading spaces into tokens (e.g., `▁Queen` → chars `(48, 54)` instead of `(49, 54)`). The decoder corrects these offsets, verified by round-trip tests across all SentencePiece checkpoints.
