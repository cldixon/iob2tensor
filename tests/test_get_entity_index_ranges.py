from iob2tensor.labels import IGNORE_TOKEN, create_label_map
from iob2tensor.checker import get_entity_index_ranges

MY_CUSTOM_LABELS = ["ARBITRARY_LABEL"]

test_cases = {
    "no_entities": {
        "example": [IGNORE_TOKEN] + [0] * 25 + [IGNORE_TOKEN],
        "target": []
    },
    "single_short_entity": {
        "example": [IGNORE_TOKEN, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0],
        "target": [(5, 7)]
    },
    "single_long_entity": {
        "example": [IGNORE_TOKEN, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, IGNORE_TOKEN],
        "target": [(3, 10)]
    },
    "multi_entities_separated": {
        "example": [IGNORE_TOKEN, 0, 0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2, 2, 0, 0, 0, IGNORE_TOKEN],
        "target": [(4, 8), (12, 16)]
    },
    "multi_entities_consecutive": {
        "example": [IGNORE_TOKEN, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 2, 0, 0, IGNORE_TOKEN],
        "target": [(4, 7), (8, 12)]
    }
}

def test_get_entity_index_ranges():
    label_map = create_label_map(labels=MY_CUSTOM_LABELS)

    # range match function tests...
    for _name, _case in test_cases.items():
        matched_range = get_entity_index_ranges(label_map, _case["example"])
        assert matched_range == _case["target"], f"Range matching failed for case '{_name}'. Returned {matched_range}, but expected {_case['target']}."
