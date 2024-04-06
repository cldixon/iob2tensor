from typing import Any
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, ValidationError
from pydantic import AfterValidator, StrictStr, StrictInt


class DefaultFields:
    TEXT = "text"
    SPANS = "spans"
    START = "start"
    END = "end"
    LABEL = "label"

## -- Pydantic models to perform validation of input annotation data
## -- and support conversion for iob-label conversion.

class _SpanFormat(BaseModel):
    start: StrictInt = Field(..., description="Index of entity starting character in text string.")
    end: StrictInt = Field(..., description="Index of entity ending character in text string.")
    label: StrictStr = Field(..., description="Label name for annotated entity (e.g., PERSON, PRODUCT, LOCATION, etc.")

class _AnnotationFormat(BaseModel):
    text: StrictStr
    spans: list[_SpanFormat]

def convert_to_validated_format(
    text: str,
    spans: list[dict],
    start_field: str,
    end_field: str,
    label_field: str
) -> _AnnotationFormat:
    """Intermediate function for converting input annotation data to pydantic models for validation and field conversion."""
    return _AnnotationFormat(
        text=text,
        spans=[
            _SpanFormat(
                start=span[start_field],
                end=span[end_field],
                label=span[label_field]
            ) for span in spans
        ]
    )

## -- typed-dicts are returned from validation step as they are _lightly_ typed
## -- but benefit from intermediate pydantic validation and preprocessing.
class Span(TypedDict):
    start: int
    end: int
    label: str

class Annotation(TypedDict):
    text: str
    spans: list[Span]

def preprocessing(
    text: str,
    spans: list[dict],
    start_field: str = DefaultFields.START,
    end_field: str = DefaultFields.END,
    label_field: str = DefaultFields.LABEL
) -> Annotation:
    # first convert to pydantic models to convert and validate input data
    validated = convert_to_validated_format(text, spans, start_field, end_field, label_field)
    # return as typed dict
    return Annotation(**validated.model_dump())

def validate_batch(
    annotations: list[dict],
    text_field: str = DefaultFields.TEXT,
    spans_field: str = DefaultFields.SPANS,
    start_field: str = DefaultFields.START,
    end_field: str = DefaultFields.END,
    label_field: str = DefaultFields.LABEL
) -> list[Annotation]:
    assert isinstance(annotations, list) and all([isinstance(ann, dict) for ann in annotations]), f"Input for annotations is not a list of dicts."
    return [
        validate(
            text=ann[text_field],
            spans=ann[spans_field],
            start_field=start_field,
            end_field=end_field,
            label_field=label_field
        )
        for ann in annotations
    ]
