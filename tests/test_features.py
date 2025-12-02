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
    t = np.arange(128) / sfreq
    X = np.sin(2 * np.pi * 6 * t).reshape(1, 2, -1)
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
    with pytest.raises(
        ValueError, match=r"^X must have shape \(n_samples, n_channels, n_times\)$"
    ):
        transformer.transform(bad_input)


def test_extract_features_fit_is_identity() -> None:
    """Verify fit returns the transformer instance for pipeline compatibility."""

    transformer = ExtractFeatures(sfreq=128.0)
    assert transformer.fit(np.zeros((1, 1, 2))) is transformer


def test_extract_features_rejects_unknown_strategy() -> None:
    """Ensure unsupported feature strategies raise explicit errors."""

    transformer = ExtractFeatures(sfreq=128.0, feature_strategy="unknown")
    with pytest.raises(
        ValueError, match=r"^Unsupported feature strategy: unknown$"
    ):
        transformer.transform(np.zeros((1, 1, 2)))


def test_extract_features_defaults_enable_normalization() -> None:
    """Confirm transformer defaults to normalized outputs."""

    transformer = ExtractFeatures(sfreq=128.0)
    assert transformer.normalize is True


def test_extract_features_respects_default_normalization() -> None:
    """Check helper defaults normalize per sample using FFT strategy."""

    sfreq = 128.0
    t = np.arange(128) / sfreq
    signals = np.stack([
        np.sin(2 * np.pi * 6 * t),
        np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 12 * t),
    ])
    X = signals.reshape(2, 1, -1)
    raw = ExtractFeatures(sfreq=sfreq, normalize=False).transform(X)
    normalized = extract_features(X, sfreq=sfreq)
    expected = (raw - raw.mean(axis=1, keepdims=True)) / (
        raw.std(axis=1, keepdims=True) + ExtractFeatures.NORMALIZATION_EPS
    )
    np.testing.assert_array_equal(normalized, expected)
    assert not np.allclose(expected[0], expected[1])


def test_extract_features_accepts_disable_normalization() -> None:
    """Ensure helper forwards normalize=False to skip scaling."""

    sfreq = 128.0
    t = np.arange(128) / sfreq
    X = np.sin(2 * np.pi * 6 * t).reshape(1, 1, -1)
    raw = ExtractFeatures(sfreq=sfreq, normalize=False).transform(X)
    features = extract_features(X, sfreq=sfreq, normalize=False)
    np.testing.assert_array_equal(features, raw)


def test_extract_features_fft_band_power_matches_reference() -> None:
    """FFT aggregation must match manual power calculation per band."""

    sfreq = 100.0
    n_times = 100
    t = np.arange(n_times) / sfreq
    signal = np.sin(2 * np.pi * 4 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    X = signal.reshape(1, 1, -1)
    transformer = ExtractFeatures(sfreq=sfreq, normalize=False)
    features = transformer.transform(X)

    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    power = np.abs(np.fft.rfft(X, axis=2)) ** 2

    def band_power(low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs <= high)
        return power[:, :, mask].mean(axis=2)

    expected = np.concatenate(
        [
            band_power(4.0, 7.0),
            band_power(8.0, 12.0),
            band_power(13.0, 30.0),
            band_power(31.0, 45.0),
        ],
        axis=1,
    )
    assert np.allclose(features, expected)
