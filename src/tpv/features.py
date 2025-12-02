"""Feature extraction utilities for EEG signals."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """Extract band power features from EEG recordings."""

    BAND_RANGES = {
        "theta": (4.0, 7.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
        "gamma": (31.0, 45.0),
    }

    EXPECTED_EEG_NDIM = 3
    NORMALIZATION_EPS = 1e-12

    def __init__(
        self, sfreq: float, feature_strategy: str = "fft", normalize: bool = True
    ):
        self.sfreq = sfreq
        self.feature_strategy = feature_strategy
        self.normalize = normalize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X.ndim != self.EXPECTED_EEG_NDIM:
            raise ValueError("X must have shape (n_samples, n_channels, n_times)")
        raw_features = self._compute_features(X)
        if self.normalize:
            mean = raw_features.mean(axis=1, keepdims=True)
            std = raw_features.std(axis=1, keepdims=True) + self.NORMALIZATION_EPS
            return (raw_features - mean) / std
        return raw_features

    @property
    def band_labels(self):
        return list(self.BAND_RANGES.keys())

    def _compute_features(self, X):
        if self.feature_strategy == "fft":
            return self._compute_fft_features(X)
        if self.feature_strategy == "wavelet":
            return np.zeros((X.shape[0], X.shape[1] * len(self.BAND_RANGES)))
        raise ValueError(f"Unsupported feature strategy: {self.feature_strategy}")

    def _compute_fft_features(self, X):
        freqs = np.fft.rfftfreq(X.shape[2], d=1.0 / self.sfreq)
        power = np.abs(np.fft.rfft(X, axis=2)) ** 2
        features = []
        for band in self.band_labels:
            low, high = self.BAND_RANGES[band]
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = power[:, :, band_mask].mean(axis=2)
            features.append(band_power)
        return np.concatenate(features, axis=1)


def extract_features(
    X, sfreq: float, feature_strategy: str = "fft", normalize: bool = True
):
    """Helper to compute features without instantiating the transformer manually."""

    transformer = ExtractFeatures(
        sfreq=sfreq, feature_strategy=feature_strategy, normalize=normalize
    )
    return transformer.transform(X)
