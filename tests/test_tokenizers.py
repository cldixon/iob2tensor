from transformers import AutoTokenizer

from iob2tensor import preprocessing, create_label_map, to_iob_tensor
from iob2tensor.tokenizers import SUPPORTED_TOKENIZERS

labels = ["actor", "character", "plot"]

example_annotation = {
    "text": "Did Dame Judy Dench star in a British film about Queen Elizabeth?",
    "spans": [
        {"label": "actor", "start": 4, "end": 19},
        {"label": "plot", "start": 30, "end": 37},
        {"label": "character", "start": 49, "end": 64}
    ]
}

def test_tokenizer_compatability():

    for checkpoint in SUPPORTED_TOKENIZERS:

        try:
            # -- initialize tokenizer ----
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            validated_annotation = preprocessing(**example_annotation)

            # -- create label index/map ----
            label_map = create_label_map(labels)

            # -- convert annotation to iob tensor format ----
            iob_tensor = to_iob_tensor(
                validated_annotation,
                label_map,
                tokenizer,
                conversion_check=True
            )
        except Exception as error:
            raise Exception(f"Compatability error for tokenizer checkpoint {checkpoint}. Encountered the following error:\n{error}.")
