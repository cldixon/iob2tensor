from itertools import takewhile

from tokenizers import Tokenizer

from iob2labels.labels import IobPrefixes, IGNORE_TOKEN
from iob2labels.annotations import Annotation, DefaultFields

def invert_label_map(label_map: dict[str, int]) -> dict[int, str]:
 return {v: k for k, v in label_map.items()}

def get_iob_type_by_iob_label(label_map: dict[str, int], iob_label: int) -> str:
    if iob_label == IGNORE_TOKEN:
        iob_type = IobPrefixes.OUTSIDE
    else:
        idx_map = invert_label_map(label_map)
        iob_type = idx_map[iob_label]
    return iob_type[0]

def is_beginning_tag(label_map: dict[str, int], iob_label: int) -> bool:
    """Boolean check if label associated with input index is a beginning tag."""
    return get_iob_type_by_iob_label(label_map, iob_label) == IobPrefixes.BEGINNING

def is_inside_tag(label_map: dict[str, int], iob_label: int) -> bool:
    """Boolean check if label associated with input index is an inside tag."""
    return get_iob_type_by_iob_label(label_map, iob_label) == IobPrefixes.INSIDE

def is_outside_tag(label_map: dict[str, int], iob_label: int) -> bool:
    """Boolean check if label associated with input index is an outside tag."""
    return get_iob_type_by_iob_label(label_map, iob_label) == IobPrefixes.OUTSIDE



def get_entity_sequence_length(label_map: dict[str, int], iob_labels: list[int]) -> int:
    return len(list(takewhile(lambda x: is_inside_tag(label_map, x), iob_labels)))

def get_entity_index_ranges(label_map: dict[str, int], iob_labels: list[int]) -> list[tuple[int, int]]:
    return [
        (idx, idx + get_entity_sequence_length(label_map, iob_labels[(idx + 1):]))
        for idx in range(len(iob_labels)) if is_beginning_tag(label_map, iob_labels[idx])
    ]

def check_iob_conversion(
    iob_labels: list[int],
    label_map: dict[str, int],
    tokenizer: Tokenizer,
    input_ids: list[int],
    annotation: Annotation,
    debug: bool = False,
    strict: bool = True,
) -> None:
    """Tests to ensure assigned IOB labels are correct based on character and token indices for annotated entities."""
    match_ranges = get_entity_index_ranges(label_map, iob_labels)
    num_ranges, num_spans = len(match_ranges), len(annotation[DefaultFields.SPANS])
    assert num_ranges == num_spans, f"Test found {num_ranges} matches but annotation includes {num_spans} entities."

    for _range, span in zip(match_ranges, annotation[DefaultFields.SPANS]):
        # recover entity from IOB positive label indices
        entity_input_ids = input_ids[_range[0]: (_range[1] + 1)]
        recovered_entity = tokenizer.decode(entity_input_ids).strip() # <- some tokenizers (e.g., Roberta-Base), can include leading whitespace in decoded entity

        # encode/decode annotated entity and assert equality
        annotated_entity = annotation[DefaultFields.TEXT][span[DefaultFields.START]:span[DefaultFields.END]]
        expected_entity = tokenizer.decode(
            tokenizer.encode(annotated_entity, add_special_tokens=False).ids  # <- standalone tokenizers: encode() returns Encoding, access .ids for token IDs
        )
        result = expected_entity == recovered_entity if strict else expected_entity in recovered_entity
        assert result, f"Recovered entity (via IOB labels) '{recovered_entity}' does not match expected entity '{annotated_entity}'. Decoded form is '{expected_entity}'."

        if debug: print(f"| -> recovered entity '{recovered_entity}' for IOB labels at indices ({_range[0]}, {_range[1]}), which matches annotated entity '{annotated_entity}'.")

    return
