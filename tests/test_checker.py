import pytest

from transformers import AutoTokenizer

from iob_labels import create_label_map
from iob_labels.annotations import preprocessing, DefaultFields
from iob_labels.checker import check_iob_conversion

## -- set constants and configurations
TOKENIZER_CHECKPOINT = "bert-base-uncased"

LABELS = ["actor", "character", "plot"]

TEST_CASES = [
    {
        "annotation": {
            "text": "Did Dame Judy Dench star in a British film about Queen Elizabeth?",
            "spans": [
                {"label": "actor", "start": 4, "end": 19},
                {"label": "plot", "start": 30, "end": 37},
                {"label": "character", "start": 49, "end": 64}
            ]
        },
        "correct_iob_labels": [-100, 0, 1, 2, 2, 2, 0, 0, 0, 5, 0, 0, 3, 4, 0, -100],
        "incorrect_iob_labels": [-100, 0, 1, 2, 2, 0, 0, 0, 0, 5, 0, 0, 3, 4, 0, -100] # <- switch concluding label '2' to '0'
    },
    {
        "annotation": {
            "text": "How many times has Matt Damon been Jason Bourne?",
            "spans": [
                {"label": "actor", "start": 19, "end": 29},
                {"label": "character", "start": 35, "end": 47}
            ]
        },
        "correct_iob_labels":  [-100, 0, 0, 0, 0, 1, 2, 0, 3, 4, 0, -100],
        "incorrect_iob_labels": [-100, 0, 0, 0, 0, 0, 2, 0, 3, 4, 0, -100] # <- removed the leadeing '1' (i.e., beginning tag)
    }
]

def test_conversion_checker():
    label_map = create_label_map(LABELS)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)

    for _case in TEST_CASES:
        annotation = _case["annotation"]
        annotation = preprocessing(**annotation)

        encoded = tokenizer(annotation[DefaultFields.TEXT], truncation=True)

        # happy path... correct conversion
        check_iob_conversion(
            _case["correct_iob_labels"],
            label_map,
            tokenizer,
            encoded["input_ids"],
            annotation
        )

        # bad path... incorrect conversion
        with pytest.raises(AssertionError) as expected_error:
            check_iob_conversion(
                _case["incorrect_iob_labels"],
                label_map,
                tokenizer,
                encoded["input_ids"],
                annotation
            )
