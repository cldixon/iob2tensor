## -- version (read from pyproject.toml at install time)
from importlib.metadata import version as _version

__version__: str = _version("iob2labels")

## -- primary interface
from .encoder import IOB2Encoder as IOB2Encoder

## -- types
from .annotations import Annotation as Annotation
from .annotations import Span as Span

## -- utilities
from .labels import create_label_map as create_label_map
from .labels import format_entity_label as format_entity_label
from .annotations import preprocessing as preprocessing

## -- checker
from .checker import check_iob_conversion as check_iob_conversion
from .checker import get_entity_index_ranges as get_entity_index_ranges
