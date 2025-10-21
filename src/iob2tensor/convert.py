import torch
from transformers import PreTrainedTokenizer

from iob2tensor.annotations import Annotation, DefaultFields
from iob2tensor.checker import check_iob_conversion
from iob2tensor.labels import (
    IGNORE_TOKEN,
    IobPrefixes,
    LabelMap,
    format_entity_label,
)


def to_iob_tensor(
    annotation: Annotation,
    label_map: LabelMap,
    tokenizer: PreTrainedTokenizer,
    ignore_token: int = IGNORE_TOKEN,
    ends_at_next_char: bool = True,
    conversion_check: bool = True,
    return_as_list: bool = False,
) -> torch.Tensor | list[int]:
    """Create target tensor from NER span annotations following IOB format. The process requires use of the
    original annotation spans and text, encoded representation of the text, and features from the Huggingface
    Tokenizer. As a result, a few things happen in this function and a dictionary of outputs are returned."""
    encoded = tokenizer(annotation[DefaultFields.TEXT], truncation=True)

    target_labels = [
        ignore_token
        if input_id in tokenizer.all_special_ids
        else label_map[IobPrefixes.OUTSIDE]
        for input_id in encoded["input_ids"]  # type: ignore
    ]

    for entity in annotation[DefaultFields.SPANS]:
        token_start = encoded.char_to_token(entity[DefaultFields.START])
        token_end = encoded.char_to_token(
            entity[DefaultFields.END] - 1
            if ends_at_next_char
            else entity[DefaultFields.END]
        )

        # get entity label values (beginning + inside)
        b_ent = label_map[
            format_entity_label(IobPrefixes.BEGINNING, entity[DefaultFields.LABEL])
        ]
        i_ent = label_map[
            format_entity_label(IobPrefixes.INSIDE, entity[DefaultFields.LABEL])
        ]

        # replace filled 'outside' label with entity labels
        target_labels = (
            target_labels[:token_start]
            + [b_ent]
            + [i_ent] * (token_end - token_start)
            + target_labels[(token_end + 1) :]
        )

    # test that final labels is same shape as input ids
    if conversion_check:
        check_iob_conversion(
            target_labels,
            label_map,
            tokenizer,
            encoded["input_ids"],  # type: ignore
            annotation,
        )
    if return_as_list is False:
        target_labels = torch.tensor(target_labels)
    return target_labels
