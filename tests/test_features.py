"""Unit tests for feature extraction utilities."""

import time

import numpy as np
import pytest

from tpv.features import ExtractFeatures, extract_features

TIME_BUDGET_S = 0.05


def test_extract_features_shape_and_labels() -> None:
    """Ensure FFT features expose correct shape and band labels."""

    sfreq = 128.0
    rng = np.random.default_rng(seed=0)
    X = rng.standard_normal((2, 3, 256))
    transformer = ExtractFeatures(sfreq=sfreq, feature_strategy="fft", normalize=True)
    features = transformer.transform(X)
    assert features.shape == (2, 12)
    assert transformer.band_labels == ["theta", "alpha", "beta", "gamma"]


def test_extract_features_normalization_and_placeholder() -> None:
    """Check normalization behavior and wavelet placeholder output."""

    sfreq = 256.0
    X = np.ones((1, 2, 128))
    normalized = extract_features(
        X, sfreq=sfreq, feature_strategy="fft", normalize=True
    )
    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-9)
    assert np.all(normalized.std(axis=1) <= 1.0 + 1e-6)
    placeholder = extract_features(
        X, sfreq=sfreq, feature_strategy="wavelet", normalize=False
    )
    assert np.array_equal(placeholder, np.zeros((1, 8)))


def test_extract_features_runtime_budget() -> None:
    """Validate latency stays under 50 ms for a small batch."""

    sfreq = 128.0
    rng = np.random.default_rng(seed=1)
    X = rng.standard_normal((5, 4, 512))
    start = time.perf_counter()
    ExtractFeatures(sfreq=sfreq).transform(X)
    stop = time.perf_counter()
    assert (stop - start) < TIME_BUDGET_S


def test_extract_features_invalid_shape() -> None:
    """Ensure invalid shapes raise explicit errors."""

    transformer = ExtractFeatures(sfreq=128.0)
    bad_input = np.zeros((10, 100))
    with pytest.raises(ValueError):
        transformer.transform(bad_input)
