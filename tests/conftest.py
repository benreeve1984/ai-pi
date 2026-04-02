import tempfile
from pathlib import Path

import pytest

from app.state import AppState


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def state(tmp_data_dir):
    return AppState(data_dir=tmp_data_dir)
