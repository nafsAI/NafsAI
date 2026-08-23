"""
conftest.py — NafsAI
Shared fixtures for all tests.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from nafsai import Agent, Cache, Memory, Normalizer, Router, Session


@pytest.fixture
def normalizer():
    return Normalizer()


@pytest.fixture
def cache():
    return Cache()


@pytest.fixture
def session():
    return Session(max_turns=5)


@pytest.fixture
def mock_model():
    """Fake model to speed up tests — no real inference."""
    model = MagicMock()

    def fake_encode(text, **kwargs):
        if isinstance(text, list):
            return np.random.rand(len(text), 768).astype("float32")
        return np.random.rand(768).astype("float32")

    model.encode = fake_encode
    return model


@pytest.fixture
def router(mock_model):
    return Router(_shared_model=mock_model)


@pytest.fixture
def memory(mock_model):
    """SQLite in-memory — no file, no server. Auto-detects available mode."""
    return Memory(db_path=":memory:", _shared_model=mock_model)


@pytest.fixture
def memory_no_vec(mock_model):
    """Forces fts_only mode — tests FTS5 fallback."""
    mem = Memory(db_path=":memory:", _shared_model=mock_model)
    mem.mode = "fts_only"
    return mem


@pytest.fixture
def memory_numpy_only(mock_model):
    """Forces numpy_only mode — tests lowest fallback."""
    mem = Memory(db_path=":memory:", _shared_model=mock_model)
    mem.mode = "numpy_only"
    return mem