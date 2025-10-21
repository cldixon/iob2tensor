from typing import Literal

IGNORE_TOKEN = -100  # <- pytorch-specific
DEFAULT_TAG_LABEL = "LABEL"


class IobPrefixes:
    OUTSIDE = "O"
    BEGINNING = "B"
    INSIDE = "I"


class IobLabelNames:
    OUTSIDE = "outside"
    BEGINNING = "beginning"
    INSIDE = "inside"


def format_entity_label(
    prefix: Literal["I", "B"], label: str = DEFAULT_TAG_LABEL
) -> str:
    assert prefix == IobPrefixes.INSIDE or prefix == IobPrefixes.BEGINNING
    return f"{prefix}-{label.upper()}"


## -- this function creates an IOB label[str] -> index[int] dictionary
## -- for mapping IOB labels (e.g., 'O', 'B-LABEL', 'I-LABEL') to integers
## -- to support NER training. Additionally, the function takes an optional
## -- list of arbitrary labels (must be strings, must be unique) and dynamically
## -- creates this dictionary by incrementing the index values accordingly.

Tag = str
Label = int
LabelMap = dict[Tag, Label]


def create_label_map(labels: list[str] | None = None) -> LabelMap:
    """Construct NER-IOB label index mapping with input list of labels."""
    if labels is None:
        labels = [DEFAULT_TAG_LABEL]
    assert len(labels) == len(
        set(labels)
    ), "Input labels contains duplicate values. Labels must be unique."
    assert all(
        [isinstance(_lbl, str) for _lbl in labels]
    ), "Input labels must contain only strings. Other type(s) detected."

    label_map = {IobPrefixes.OUTSIDE: 0}
    for i in range(0, len(labels) * 2, 2):
        label = labels[i // 2].upper()
        label_map.update(
            {
                format_entity_label(prefix=IobPrefixes.BEGINNING, label=label): (
                    i + 1
                ),  # e.g., "B-ORG: 2"
                format_entity_label(prefix=IobPrefixes.INSIDE, label=label): (
                    i + 2
                ),  # e.g., "I-ORG: 3"
            }
        )
    return label_map
