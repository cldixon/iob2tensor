# API Reference

## IOB2Encoder

The main interface for encoding and decoding IOB2 NER annotations.

```python
from iob2labels import IOB2Encoder
```

### Constructor

```python
IOB2Encoder(
    labels: list[str],
    tokenizer: str | Tokenizer,
    *,
    ignore_token: int = -100,
    ends_at_next_char: bool = True,
    conversion_check: bool = True,
    max_length: int | None = 512,
    start_field: str = "start",
    end_field: str = "end",
    label_field: str = "label",
    text_field: str = "text",
    spans_field: str = "spans",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | `list[str]` | *required* | Entity class names (e.g., `["actor", "character", "plot"]`). Each generates `B-` and `I-` IOB2 tags. |
| `tokenizer` | `str \| Tokenizer` | *required* | HuggingFace checkpoint name, `tokenizers.Tokenizer`, or `transformers.PreTrainedTokenizerFast`. See [Tokenizer Input](guide.md#the-tokenizer-parameter). |
| `ignore_token` | `int` | `-100` | Label value assigned to special tokens. PyTorch's `CrossEntropyLoss` ignores this by default. |
| `ends_at_next_char` | `bool` | `True` | Whether span `end` offsets point to the character *after* the last entity character (standard convention). |
| `conversion_check` | `bool` | `True` | Run a round-trip verification after each encoding to catch misalignment bugs. |
| `max_length` | `int \| None` | `512` | Maximum token sequence length. Enables truncation on the tokenizer. Set to `None` to disable. |
| `start_field` | `str` | `"start"` | Key for start offset in span dicts. |
| `end_field` | `str` | `"end"` | Key for end offset in span dicts. |
| `label_field` | `str` | `"label"` | Key for entity label in span dicts. |
| `text_field` | `str` | `"text"` | Key for text string in batch annotation dicts. |
| `spans_field` | `str` | `"spans"` | Key for spans list in batch annotation dicts. |

### Properties

#### `label_map` -> `dict[str, int]`

The IOB2 label-to-index mapping. Returns a copy to prevent mutation.

```python
encoder.label_map
# {'O': 0, 'B-ACTOR': 1, 'I-ACTOR': 2, 'B-CHARACTER': 3, 'I-CHARACTER': 4, 'B-PLOT': 5, 'I-PLOT': 6}
```

#### `tokenizer` -> `Tokenizer`

The resolved `tokenizers.Tokenizer` instance.

### Methods

#### `__call__(text, spans)` -> `list[int]`

Encode a single annotation into IOB2 label indices.

```python
labels = encoder(
    text="Did Dame Judy Dench star?",
    spans=[{"label": "actor", "start": 4, "end": 19}],
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The raw input text. |
| `spans` | `list[dict]` | Span dicts with start/end/label fields (field names configurable via constructor). |

**Returns:** `list[int]` of IOB2 labels aligned to tokenizer output.

---

#### `batch(annotations, *, on_error="raise")` -> `list[list[int]]`

Encode a batch of annotations. Uses Rust-backed `encode_batch()` for parallelized tokenization.

```python
results = encoder.batch(annotations, on_error="skip")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `annotations` | `list[dict]` | *required* | Annotation dicts with text/spans fields. |
| `on_error` | `str` | `"raise"` | `"raise"` to fail on first error, `"skip"` to skip failed annotations. |

**Returns:** `list[list[int]]` of IOB2 label sequences (unpadded).

---

#### `decode(labels, encoding, text)` -> `list[Span]`

Recover span annotations from IOB2 label indices given a pre-built `Encoding` object.

```python
encoding = encoder.tokenizer.encode(text)
spans = encoder.decode(predicted_labels, encoding, text)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `labels` | `list[int]` | IOB2 label indices (e.g., model predictions after argmax). |
| `encoding` | `Encoding` | The `tokenizers.Encoding` object for the text. |
| `text` | `str` | The original text (needed to resolve SentencePiece whitespace boundaries). |

**Returns:** `list[Span]` with character-level `start`, `end`, and `label` fields.

---

#### `decode_text(labels, text)` -> `list[Span]`

Convenience method: tokenizes the text internally, then decodes.

```python
spans = encoder.decode_text(predicted_labels, text)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `labels` | `list[int]` | IOB2 label indices. |
| `text` | `str` | The raw input text (will be tokenized internally). |

**Returns:** `list[Span]` with character-level `start`, `end`, and `label` fields.

---

## Types

### Span

A `TypedDict` representing a single entity annotation with character offsets.

```python
from iob2labels import Span
```

| Field | Type | Description |
|-------|------|-------------|
| `start` | `int` | Start character offset (inclusive). |
| `end` | `int` | End character offset (exclusive). |
| `label` | `str` | Entity class name. |

### Annotation

A `TypedDict` representing a text with its entity annotations.

```python
from iob2labels import Annotation
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The annotated text. |
| `spans` | `list[Span]` | Entity annotations. |

---

## Utility Functions

### `create_label_map(labels)` -> `dict[str, int]`

Build an IOB2 label-to-index mapping from a list of entity class names.

```python
from iob2labels import create_label_map

label_map = create_label_map(["actor", "character"])
# {'O': 0, 'B-ACTOR': 1, 'I-ACTOR': 2, 'B-CHARACTER': 3, 'I-CHARACTER': 4}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | `list[str] \| None` | `None` | Entity class names. Defaults to `["LABEL"]` if `None`. |

---

### `format_entity_label(prefix, label)` -> `str`

Format an IOB2 entity label string.

```python
from iob2labels import format_entity_label

format_entity_label("B", "actor")
# 'B-ACTOR'
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefix` | `"B" \| "I"` | IOB2 prefix (Beginning or Inside). |
| `label` | `str` | Entity class name. |

---

### `preprocessing(text, spans, ...)` -> `Annotation`

Validate and normalize annotation data via Pydantic, then return as a typed dict.

```python
from iob2labels import preprocessing

annotation = preprocessing(
    text="Hello world",
    spans=[{"start": 0, "end": 5, "label": "greeting"}],
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | The input text. |
| `spans` | `list[dict]` | *required* | Span dicts to validate. |
| `start_field` | `str` | `"start"` | Key for start offset. |
| `end_field` | `str` | `"end"` | Key for end offset. |
| `label_field` | `str` | `"label"` | Key for entity label. |

**Raises:** `ValidationError` for type mismatches, `ValueError` for invalid span geometry.

---

### `check_iob_conversion(...)` -> `None`

Verify that encoded IOB2 labels correctly recover the original entity text. Used internally when `conversion_check=True`.

```python
from iob2labels import check_iob_conversion

check_iob_conversion(
    iob_labels=labels,
    label_map=encoder.label_map,
    tokenizer=encoder.tokenizer,
    input_ids=encoding.ids,
    annotation=annotation,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `iob_labels` | `list[int]` | The encoded label sequence to verify. |
| `label_map` | `dict[str, int]` | IOB2 label-to-index mapping. |
| `tokenizer` | `Tokenizer` | The tokenizer instance. |
| `input_ids` | `list[int]` | Token IDs from the encoding. |
| `annotation` | `Annotation` | The original annotation for comparison. |

**Raises:** `AssertionError` if the recovered entities do not match the original spans.

---

### `get_entity_index_ranges(label_map, iob_labels)` -> `list[tuple[int, int]]`

Extract token index ranges for each entity from an IOB2 label sequence.

```python
from iob2labels import get_entity_index_ranges

ranges = get_entity_index_ranges(encoder.label_map, labels)
# [(1, 4), (8, 8), (11, 12)]  — (start_token_idx, end_token_idx) for each entity
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `label_map` | `dict[str, int]` | IOB2 label-to-index mapping. |
| `iob_labels` | `list[int]` | The IOB2 label sequence to scan. |
