"""Unit tests for feature extraction utilities."""

# Import time to benchmark feature extraction speed
import time

# Import numpy to build deterministic dummy EEG inputs
import numpy as np

# Import pytest to validate expected failures and shapes
import pytest

# Import the feature transformer to exercise band aggregation
from tpv.features import ExtractFeatures, extract_features


def test_extract_features_shape_and_labels() -> None:
    """Ensure FFT features expose correct shape and band labels."""

    # Set sampling frequency to map FFT bins consistently during tests
    sfreq = 128.0
    # Create deterministic input with two samples, three channels, and short duration
    rng = np.random.default_rng(seed=0)
    # Generate synthetic EEG data to keep tests reproducible
    X = rng.standard_normal((2, 3, 256))
    # Instantiate the transformer with default FFT strategy
    transformer = ExtractFeatures(sfreq=sfreq, feature_strategy="fft", normalize=True)
    # Compute normalized features for the synthetic batch
    features = transformer.transform(X)
    # Expect four bands times three channels in the flattened output
    assert features.shape == (2, 12)
    # Verify the band labels preserve the canonical order
    assert transformer.band_labels == ["theta", "alpha", "beta", "gamma"]


def test_extract_features_normalization_and_placeholder() -> None:
    """Check normalization behavior and wavelet placeholder output."""

    # Set sampling frequency to anchor FFT calculations during comparison
    sfreq = 256.0
    # Build a single-sample batch to probe normalization stability
    X = np.ones((1, 2, 128))
    # Compute normalized FFT features to validate zero mean and unit variance
    normalized = extract_features(X, sfreq=sfreq, feature_strategy="fft", normalize=True)
    # Confirm per-sample means are numerically close to zero after normalization
    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-9)
    # Confirm per-sample standard deviations remain finite and bounded by one
    assert np.all(normalized.std(axis=1) <= 1.0 + 1e-6)
    # Compute placeholder wavelet features to ensure shapes remain stable
    placeholder = extract_features(X, sfreq=sfreq, feature_strategy="wavelet", normalize=False)
    # Expect zeros because the wavelet path is not implemented yet
    assert np.array_equal(placeholder, np.zeros((1, 8)))


def test_extract_features_runtime_budget() -> None:
    """Validate latency stays under 50 ms for a small batch."""

    # Configure sampling frequency to align with EEG acquisition defaults
    sfreq = 128.0
    # Generate a batch with modest size to benchmark runtime without noise
    rng = np.random.default_rng(seed=1)
    # Construct five samples with four channels to stress the loop lightly
    X = rng.standard_normal((5, 4, 512))
    # Record start time to measure end-to-end feature extraction latency
    start = time.perf_counter()
    # Execute the FFT-based transformation under test
    ExtractFeatures(sfreq=sfreq).transform(X)
    # Record stop time immediately after transformation
    stop = time.perf_counter()
    # Ensure the elapsed time respects the lightweight budget
    assert (stop - start) < 0.05


def test_extract_features_invalid_shape() -> None:
    """Ensure invalid shapes raise explicit errors."""

    # Instantiate the transformer with a valid sampling frequency
    transformer = ExtractFeatures(sfreq=128.0)
    # Provide a malformed input missing the channel dimension
    bad_input = np.zeros((10, 100))
    # Expect a ValueError when the input shape is not three-dimensional
    with pytest.raises(ValueError):
        transformer.transform(bad_input)
