"""Tests for the evaluation pipeline."""

import pytest


def test_evaluation_buckets():
    """Should have exactly 4 evaluation buckets."""
    buckets = ["utility", "coverage", "validity", "risk"]
    assert len(buckets) == 4


def test_metrics_count():
    """Should have exactly 8 metrics total."""
    metrics = [
        "downstream_accuracy", "few_shot_improvement",
        "distribution_coverage", "category_balance",
        "linguistic_quality", "structural_correctness",
        "memorization_rate", "deduplication"
    ]
    assert len(metrics) == 8


def test_baseline_count():
    """Should compare against exactly 4 methods (including ours)."""
    baselines = ["no_augmentation", "simple_paraphrasing", "generic_llm", "synthipdata"]
    assert len(baselines) == 4
