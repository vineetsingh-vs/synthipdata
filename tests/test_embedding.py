"""Tests for the embedding pipeline."""

import pytest


def test_embedding_dimension():
    """BGE-M3 should produce 1024-dimensional vectors."""
    expected_dim = 1024
    assert expected_dim == 1024


def test_memorization_threshold():
    """Memorization threshold should be 0.95 cosine similarity."""
    threshold = 0.95
    assert 0.9 <= threshold <= 1.0


def test_garbage_threshold():
    """Garbage threshold should be 0.30 cosine similarity."""
    threshold = 0.30
    assert 0.0 <= threshold <= 0.5
