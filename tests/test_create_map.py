from iob2labels import create_label_map

TEST_LABELS = ["character", "actor", "plot"]

TEST_TARGET_WITH_LABELS = {
    "O": 0,
    "B-CHARACTER": 1,
    "I-CHARACTER": 2,
    "B-ACTOR": 3,
    "I-ACTOR": 4,
    "B-PLOT": 5,
    "I-PLOT": 6
}

TEST_TARGET_WITH_NO_LABELS = {
    "O": 0,
    "B-LABEL": 1,
    "I-LABEL": 2
}


def test_construct_label_index_with_labels():
    label_index = create_label_map(TEST_LABELS)
    assert label_index == TEST_TARGET_WITH_LABELS, f"Created label index does not match target. Instead, got: {label_index}."

def test_construct_label_index_without_labels():
    label_index = create_label_map()
    assert label_index == TEST_TARGET_WITH_NO_LABELS, f"Created label index (with no labels) does not match target. Instead, got: {label_index}."
