# Supported Tokenizers

`iob2labels` is tested across four tokenizer families covering 18 checkpoints. Both encoding and decoding are verified via round-trip tests: encode spans to labels, decode labels back to spans, and assert the recovered spans exactly match the originals.

## Tested Checkpoints

| Family | Checkpoints |
|--------|-------------|
| WordPiece | `bert-base-cased`, `bert-base-uncased`, `bert-large-cased`, `bert-large-uncased`, `bert-base-multilingual-cased`, `distilbert-base-cased`, `distilbert-base-uncased`, `google/electra-base-discriminator` |
| BPE (byte-level) | `roberta-base`, `roberta-large`, `distilroberta-base`, `allenai/longformer-base-4096` |
| SentencePiece BPE | `FacebookAI/xlm-roberta-base`, `almanach/camembert-base` |
| SentencePiece Unigram | `albert-base-v2`, `xlnet-base-cased`, `t5-small`, `google/flan-t5-base` |

## Tokenizer Families

### WordPiece

WordPiece tokenizers (BERT, DistilBERT, ELECTRA) split words into subword units prefixed with `##`. For example, "tokenization" might become `["token", "##ization"]`. Character-to-token alignment is straightforward — each subword maps cleanly to a contiguous character range.

These tokenizers are fully compatible with `iob2labels` and require no special handling.

### BPE (Byte-Level)

Byte-level BPE tokenizers (RoBERTa, Longformer) operate on byte-level representations and use `\u0120` (Ġ) as a word-boundary prefix. For example, " Queen" becomes `["ĠQueen"]`.

The `Encoding.char_to_token()` and `Encoding.token_to_chars()` methods handle the byte-level mapping correctly, so `iob2labels` works with these tokenizers without modification.

### SentencePiece BPE

SentencePiece BPE tokenizers (XLM-RoBERTa, CamemBERT) use the `▁` (U+2581) character to represent word boundaries. For example, " Queen" becomes `["▁Queen"]`.

These tokenizers absorb the leading whitespace into the token, which means `token_to_chars()` returns a range that includes the space character. The decoder handles this automatically — see [Whitespace Handling](#sentencepiece-whitespace-handling) below.

### SentencePiece Unigram

SentencePiece Unigram tokenizers (ALBERT, XLNet, T5, Flan-T5) also use the `▁` prefix convention and share the same whitespace absorption behavior as SentencePiece BPE.

!!! note "Google Flan-T5 sentinel tokens"
    Flan-T5 uses sentinel tokens (e.g., `<extra_id_0>`) in its vocabulary. The round-trip check accounts for these during conversion verification.

## SentencePiece Whitespace Handling

SentencePiece-based tokenizers (both BPE and Unigram families) absorb leading whitespace into the first token of each word. This means `token_to_chars()` can return character offsets that include the space *before* the entity rather than just the entity text.

For example, given the text `"film about Queen Elizabeth"` and an entity at `(11, 26)` covering `"Queen Elizabeth"`:

- A WordPiece tokenizer maps `"Queen"` to chars `(11, 16)` — correct
- A SentencePiece tokenizer maps `"▁Queen"` to chars `(10, 16)` — includes the preceding space

The decoder corrects this automatically by stripping leading (and trailing) whitespace from the recovered character range using the original text. No configuration needed.

## Using Untested Tokenizers

Any HuggingFace-compatible tokenizer with a `tokenizer.json` file on the Hub should work with `iob2labels`. When you use a checkpoint not in the tested list, a `UserWarning` is emitted:

```
UserWarning: Tokenizer 'my-custom/tokenizer' is not in the list of checkpoints tested
with iob2labels. It may work correctly, but results have not been verified.
```

The built-in conversion check (`conversion_check=True`, the default) will catch any alignment issues at encoding time. If the round-trip verification passes, the tokenizer is working correctly for your data.

To suppress the warning once you've verified correctness:

```python
import warnings
warnings.filterwarnings("ignore", message="Tokenizer.*not in the list")
```
