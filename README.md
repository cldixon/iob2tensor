# IOB Label Conversion for Named Entity Recognition (NER) Tasks

This repo contains simple functions for converting [IOB2-format](<https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)>) NER annotation data into tensor formats for Transformer-based NER tasks. Open source examples of this format include this [news-headlines](https://raw.githubusercontent.com/explosion/prodigy-recipes/master/example-datasets/annotated_news_headlines-ORG-PERSON-LOCATION-ner.jsonl) dataset (e.g., referenced by [Prodigy](https://prodi.gy/docs#first-steps3)) and the [biomed-ner dataset](https://huggingface.co/datasets/knowledgator/biomed_NER).

_Note:_ If you use Prodigy to annotate data for an NER task, the IOB2 format is what will be output.

_Note:_ The below functions only convert one text example at a time, so a batch job will require some additional looping, etc.

## Example

Below is an example of an NER/IOB2 format annotation:

```python
# example annotation and labels
annotation = {
    "text": "Did Dame Judy Dench star in a British film about Queen Elizabeth?",
    "spans": [
        {"label": "actor", "start": 4, "end": 19},
        {"label": "plot", "start": 30, "end": 37},
        {"label": "character", "start": 49, "end": 64}
    ]
}
```

> example pulled from [MITMovie](https://groups.csail.mit.edu/sls/downloads/movie/) dataset

In order to train a an NER model (_a la_ [Token Classification](https://huggingface.co/docs/transformers/en/tasks/token_classification) style task), we can represent the _target_ output of the above example as follows:

```python
[0, 1, 2, 2, 2, 0, 0, 0, 5, 0, 0, 3, 4, 0]
```

In which the target labels correspond to the following classes:

```sh
0 -> outside (i.e., no labels)
1,2 -> actor (beginning and inside)
3,4 -> character (beginning and inside)
5 -> plot (only beginning; word only requires 1 token)
```

The following contains instructions for producing this conversion.

## Usage

### Preprocessing and Schema Validation

One of the first challenges in preprocessing data annotated for an NER task is managing the complexity of nested annotations, different field names, and the various ways to _label_ an annotated entity. Since Transformers are coupled to a Tokenizer, an NER schema based around attaching labels to tokens (e.g., words) introduces complexity because the labeled tokens will have to be converted for every different tokenizer. Additionally, this also allows any nuances of the tokenizer used in annotation to mix with the data.

For these reasons, assigning entity labels to string indices is more generic, decoupled from any specific tokenizer, and more easily checkable for errors in the data or any subsequent processing.

The example below is pulled from the [MITMovie](https://groups.csail.mit.edu/sls/downloads/movie/) annotated dataset.

```python
# example annotation and labels
annotation = {
    "text": "Did Dame Judy Dench star in a British film about Queen Elizabeth?",
    "spans": [
        {"label": "actor", "start": 4, "end": 19},
        {"label": "plot", "start": 30, "end": 37},
        {"label": "character", "start": 49, "end": 64}
    ]
}
```

Due to the complex structure of NER spans and the associated text field, we perform a preprocessing and validation step to ensure everything is in good order. This happens thanks to [Pydantic](https://docs.pydantic.dev/latest/) as an intermediate step, but the outputs are still typed dictionaries to keep things simple for the user.

```python
from iob2tensor import preprocess

text = "Did Dame Judy Dench star in a British film about Queen Elizabeth?"

spans = [
    {"label": "actor", "start": 4, "end": 19},
    {"label": "plot", "start": 30, "end": 37},
    {"label": "character", "start": 49, "end": 64}
]
# validate input annotations
annotation = preprocess(text, spans)
```

The default or expected fields for input annotations are as follows:

```python
from typing import TypedDict

class Span(TypedDict):
    start: int
    end: int
    label: str

class Annotation(TypedDict):
    text: str
    spans: list[Span]
```

If your annotated data uses different fields, specify those fields as function arguments. For instance, the [BioMed-NER](https://huggingface.co/datasets/knowledgator/biomed_NER) dataset follows the standard NER spans schema but uses different field names.

```python
annotation = {
    "text": "Weed seed inactivation in soil mesocosms via biosolarization..." "entities": [
        {"start": 0, "end": 4, "class": "ORGANISM"},
        {"start": 5, "end": 9, "class": "ORGANISM"},
        {"start": 26, "end": 30, "class": "CHEMICALS"},
        ...
}

annotation = preprocess(
    **annotation,
    spans_field="entities",
    label_field="class"
)
```

### Create Label Map

Next, create the IOB label map with your dataset's entity labels. The default label in the IOB2 format, represents all tokens which are not entities and thus is referred to as the _outside_ class. The convention is to assign all tokens of this class as `label=0`. Additionally, the IOB2 format distinguishes between the _beginning_ of _inside_ of an entity label, so each entity class will generate 2 _distinct_ labels, following this format:

- **B-LABEL**
- **I-LABEL**

This means the label set and mapping will always have a size of `(_n_ * 2) + 1`, where _n_ equals the number of distinct labels (e.g., "location", "organization", "person", etc.) and the `+1` is from the _outside_ (non-entity) class.

Use the following function to create the initial label map for your dataset's labels.

```python
from iob2tensor import create_label_map

labels = ["actor", "character", "plot"]

label_map = create_label_map(labels)
label_map
>>> {
    'O': 0,
    'B-ACTOR': 1, 'I-ACTOR': 2,
    'B-CHARACTER': 3, 'I-CHARACTER': 4,
    'B-PLOT': 5, 'I-PLOT': 6
}
```

### Create Target Output

Now we select and initialize a tokenizer - which has to be involved in the iob label conversion due to tokenization - and convert our NER annotation into a label array.

There is a built-in conversion check (on by default) which ensures the conversion is correct. This is guaranteed to work for the supported tokenizers, but can also be turned off in order to reduce computation.

```python
from transformers import AutoTokenizer

from iob2tensor import to_iob_tensor

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

iob_labels = to_iob_tensor(annotation, label_map, tokenizer)
iob_labels
>>> [-100, 0, 1, 2, 2, 2, 0, 0, 0, 5, 0, 0, 3, 4, 0, -100]
```

Now just one step away from a tensor!

```python
import torch

x = torch.tensor(iob_labels)
```

### Tests

There is a built-in check (can be optionally turned off) within the main `to_iob_tensor()` function, which attempts to confirm the iob2 conversion is correct. Additionally, there are a series of additional unit and end-to-end tests in the `tests` directory. Finally, the `tokenizers.py` file contains the specific tokenizer checkpoints which I have tested.
