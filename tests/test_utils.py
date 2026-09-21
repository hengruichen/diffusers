import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    is_xformers_available,
    is_xformers_version,
)


@pytest.mark.skipif(
    not (is_accelerate_available() and is_accelerate_version(">=", "0.17.0")),
    reason="This test requires accelerate v0.17.0 or higher.",
)
def test_accelerate_version():
    assert is_accelerate_available()
    assert is_accelerate_version(">=", "0.17.0")


@pytest.mark.skipif(
    not (is_transformers_available() and is_transformers_version(">=", "4.28.0")),
    reason="This test requires transformers v4.28.0 or higher.",
)
def test_transformers_version():
    assert is_transformers_available()
    assert is_transformers_version(">=", "4.28.0")


@pytest.mark.skipif(
    not (is_torch_version(">=", "1.9.0")),
    reason="This test requires torch v1.9.0 or higher.",
)
def test_torch_version():
    assert is_torch_version(">=", "1.9.0")


@pytest.mark.skipif(
    not (is_xformers_available() and is_xformers_version(">=", "0.0.16")),
    reason="This test requires xformers v0.0.16 or higher.",
)
def test_xformers_version():
    assert is_xformers_available()
    assert is_xformers_version(">=", "0.0.16")


def test_tmp_dir():
    with TemporaryDirectory() as tmpdirname:
        assert os.path.isdir(tmpdirname)
        assert Path(tmpdirname).is_dir()
