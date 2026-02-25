import pytest
from tokenizers import Tokenizer

## -- tokenizer checkpoints organized by family
WORDPIECE_CHECKPOINTS = [
    "bert-base-uncased",
    "bert-base-cased",
    "bert-base-multilingual-cased",
    "distilbert-base-uncased",
    "google/electra-base-discriminator",
]

BPE_CHECKPOINTS = [
    "roberta-base",
    "distilroberta-base",
    "allenai/longformer-base-4096",
]

SENTENCEPIECE_BPE_CHECKPOINTS = [
    "FacebookAI/xlm-roberta-base",
    "almanach/camembert-base",
]

SENTENCEPIECE_UNIGRAM_CHECKPOINTS = [
    "albert-base-v2",
    "xlnet-base-cased",
    "t5-small",
    "google/flan-t5-base",
]

ALL_CHECKPOINTS = (
    WORDPIECE_CHECKPOINTS
    + BPE_CHECKPOINTS
    + SENTENCEPIECE_BPE_CHECKPOINTS
    + SENTENCEPIECE_UNIGRAM_CHECKPOINTS
)

## -- shared labels used across tests
LABELS = ["actor", "character", "plot"]

## -- standard test annotations (MITMovie examples)
STANDARD_ANNOTATION = {
    "text": "Did Dame Judy Dench star in a British film about Queen Elizabeth?",
    "spans": [
        {"label": "actor", "start": 4, "end": 19},
        {"label": "plot", "start": 30, "end": 37},
        {"label": "character", "start": 49, "end": 64}
    ]
}

MULTI_ANNOTATION = {
    "text": "How many times has Matt Damon been Jason Bourne?",
    "spans": [
        {"label": "actor", "start": 19, "end": 29},
        {"label": "character", "start": 35, "end": 47}
    ]
}

## -- edge case annotations for stress testing
EDGE_CASE_ANNOTATIONS = {
    "entity_at_start": {
        "text": "Matt Damon starred in The Bourne Identity.",
        "spans": [{"label": "actor", "start": 0, "end": 10}]
    },
    "entity_at_end": {
        "text": "The movie starred Matt Damon",
        "spans": [{"label": "actor", "start": 18, "end": 28}]
    },
    "adjacent_entities": {
        "text": "Matt Damon Jason Bourne are great",
        "spans": [
            {"label": "actor", "start": 0, "end": 10},
            {"label": "character", "start": 11, "end": 23},
        ]
    },
    "entity_with_punctuation": {
        "text": "Did you see Dr. No starring Sean Connery?",
        "spans": [
            {"label": "character", "start": 12, "end": 18},
            {"label": "actor", "start": 28, "end": 40},
        ]
    },
    "no_entities": {
        "text": "This is a movie about nothing in particular.",
        "spans": []
    },
}
