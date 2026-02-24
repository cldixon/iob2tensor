from __future__ import annotations

from tokenizers import Tokenizer, Encoding

from iob2labels.annotations import (
    Annotation,
    DefaultFields,
    preprocessing,
    validate_batch,
)
from iob2labels.checker import check_iob_conversion
from iob2labels.labels import (
    IGNORE_TOKEN,
    IobPrefixes,
    LabelMap,
    create_label_map,
    format_entity_label,
)
from iob2labels.tokenizers import resolve_tokenizer, DEFAULT_MAX_LENGTH


class IOB2Encoder:
    """Converts IOB2 NER span annotations into integer label sequences aligned to tokenizer output.

    Accepts text + span annotations, tokenizes the text, and produces a list[int] of IOB2 labels
    aligned to the token positions. Special tokens receive the ignore_token value (-100 by default).
    """

    def __init__(
        self,
        labels: list[str],
        tokenizer: str | Tokenizer,
        *,
        ignore_token: int = IGNORE_TOKEN,
        ends_at_next_char: bool = True,
        conversion_check: bool = True,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        ## -- field name remapping (per-dataset config)
        start_field: str = DefaultFields.START,
        end_field: str = DefaultFields.END,
        label_field: str = DefaultFields.LABEL,
        text_field: str = DefaultFields.TEXT,
        spans_field: str = DefaultFields.SPANS,
    ) -> None:
        self._label_map: LabelMap = create_label_map(labels)
        self._tokenizer: Tokenizer = resolve_tokenizer(tokenizer)
        self._ignore_token = ignore_token
        self._ends_at_next_char = ends_at_next_char
        self._conversion_check = conversion_check
        self._max_length = max_length

        ## -- field remapping
        self._start_field = start_field
        self._end_field = end_field
        self._label_field = label_field
        self._text_field = text_field
        self._spans_field = spans_field

        ## -- configure truncation on the tokenizer instance
        if max_length is not None:
            self._tokenizer.enable_truncation(max_length=max_length)
        else:
            self._tokenizer.no_truncation()

    ## -- public properties

    @property
    def label_map(self) -> LabelMap:
        """The IOB2 label-to-index mapping (e.g., {'O': 0, 'B-ACTOR': 1, ...})."""
        return dict(self._label_map)  # <- return a copy to prevent mutation

    @property
    def tokenizer(self) -> Tokenizer:
        """The resolved tokenizers.Tokenizer instance."""
        return self._tokenizer

    ## -- single-example encoding

    def __call__(
        self,
        text: str,
        spans: list[dict],
    ) -> list[int]:
        """Encode a single annotation into IOB2 label indices.

        Args:
            text: The raw input text.
            spans: List of span dicts, each with start/end/label fields
                   (field names configurable via constructor).

        Returns:
            list[int] of IOB2 labels aligned to tokenizer output.
        """
        ## -- validate and normalize input via pydantic
        annotation: Annotation = preprocessing(
            text=text,
            spans=spans,
            start_field=self._start_field,
            end_field=self._end_field,
            label_field=self._label_field,
        )

        ## -- encode text and build label sequence
        encoded: Encoding = self._tokenizer.encode(annotation[DefaultFields.TEXT])
        return self._encode_single(annotation, encoded)

    ## -- batch encoding

    def batch(
        self,
        annotations: list[dict],
        *,
        on_error: str = "raise",
    ) -> list[list[int]]:
        """Encode a batch of annotations into IOB2 label sequences.

        Args:
            annotations: List of annotation dicts, each with text/spans fields
                         (field names configurable via constructor).
            on_error: Error handling strategy.
                'raise' (default) - raise on first error.
                'skip' - skip failed annotations, return results for successful ones.

        Returns:
            list[list[int]] of IOB2 label sequences.
        """
        assert on_error in ("raise", "skip"), (
            f"on_error must be 'raise' or 'skip', got '{on_error}'."
        )

        ## -- validate all annotations upfront
        validated: list[Annotation] = validate_batch(
            annotations,
            text_field=self._text_field,
            spans_field=self._spans_field,
            start_field=self._start_field,
            end_field=self._end_field,
            label_field=self._label_field,
        )

        ## -- batch-encode texts for performance (parallelized in Rust)
        texts = [ann[DefaultFields.TEXT] for ann in validated]
        encodings: list[Encoding] = self._tokenizer.encode_batch(texts)

        ## -- build label sequences for each annotation
        results: list[list[int]] = []
        for annotation, encoded in zip(validated, encodings):
            try:
                target_labels = self._encode_single(annotation, encoded)
                results.append(target_labels)
            except Exception:
                if on_error == "raise":
                    raise
                # <- on_error == "skip": silently skip this annotation

        return results

    ## -- internal encoding logic (shared between __call__ and batch)

    def _encode_single(
        self,
        annotation: Annotation,
        encoded: Encoding,
    ) -> list[int]:
        """Build IOB2 labels for a pre-validated, pre-encoded annotation."""

        ## -- build initial label sequence: ignore_token for specials, O for content tokens
        target_labels: list[int] = [
            self._ignore_token if is_special else self._label_map[IobPrefixes.OUTSIDE]
            for is_special in encoded.special_tokens_mask
        ]

        ## -- assign entity labels based on character-to-token mapping
        for entity in annotation[DefaultFields.SPANS]:
            token_start = encoded.char_to_token(entity[DefaultFields.START])
            token_end = encoded.char_to_token(
                entity[DefaultFields.END] - 1
                if self._ends_at_next_char
                else entity[DefaultFields.END]
            )

            # <- entity may be beyond truncation boundary
            if token_start is None or token_end is None:
                continue

            ## -- get entity label values (beginning + inside)
            b_ent = self._label_map[
                format_entity_label(IobPrefixes.BEGINNING, entity[DefaultFields.LABEL])
            ]
            i_ent = self._label_map[
                format_entity_label(IobPrefixes.INSIDE, entity[DefaultFields.LABEL])
            ]

            ## -- replace filled 'outside' labels with entity labels
            target_labels = (
                target_labels[:token_start]
                + [b_ent]
                + [i_ent] * (token_end - token_start)
                + target_labels[(token_end + 1):]
            )

        ## -- conversion check
        if self._conversion_check:
            check_iob_conversion(
                target_labels,
                self._label_map,
                self._tokenizer,
                encoded.ids,
                annotation,
            )

        return target_labels
