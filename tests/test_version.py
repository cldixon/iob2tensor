import iob2labels


def test_version_attribute():
    """Package exposes a __version__ string pulled from installed metadata."""
    assert hasattr(iob2labels, "__version__")
    assert isinstance(iob2labels.__version__, str)
    assert len(iob2labels.__version__) > 0
