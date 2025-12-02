# Documente le module pour cadrer les fonctions de caractéristiques EEG
"""Feature extraction utilities for EEG signals."""
#
# Import BaseEstimator to integrate with sklearn pipelines
from sklearn.base import BaseEstimator, TransformerMixin
# Import numpy to compute spectral features and normalization
import numpy as np


class ExtractFeatures(BaseEstimator, TransformerMixin):
    """Extract band power features from EEG recordings."""

    # Define EEG band boundaries to standardize power aggregation
    BAND_RANGES = {
        "theta": (4.0, 7.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
        "gamma": (31.0, 45.0),
    }

    def __init__(self, sfreq: float, feature_strategy: str = "fft", normalize: bool = True):
        # Store sampling frequency to map frequency bins to Hertz
        self.sfreq = sfreq
        # Select the feature computation strategy requested by the caller
        self.feature_strategy = feature_strategy
        # Toggle per-sample normalization to stabilize downstream models
        self.normalize = normalize

    def fit(self, X, y=None):
        # Return the transformer unchanged because no training state is required
        return self

    def transform(self, X):
        # Validate that input signals follow the expected 3D shape
        if X.ndim != 3:
            # Raise an explicit error when the input is not shaped as samples × channels × time
            raise ValueError("X must have shape (n_samples, n_channels, n_times)")
        # Compute features using the requested strategy to keep the interface extensible
        raw_features = self._compute_features(X)
        # Apply per-sample normalization when requested to equalize scales
        if self.normalize:
            # Center features per sample to stabilize mean magnitude
            mean = raw_features.mean(axis=1, keepdims=True)
            # Compute standard deviation per sample with numerical guard
            std = raw_features.std(axis=1, keepdims=True) + 1e-12
            # Normalize features to zero mean and unit variance per sample
            return (raw_features - mean) / std
        # Return raw features when normalization is disabled for diagnostics
        return raw_features

    @property
    def band_labels(self):
        # Expose band labels in a deterministic order for downstream consumers
        return list(self.BAND_RANGES.keys())

    def _compute_features(self, X):
        # Route computation to FFT or wavelet placeholder depending on configuration
        if self.feature_strategy == "fft":
            # Use FFT-based power aggregation as the default strategy
            return self._compute_fft_features(X)
        # Fall back to a placeholder path when wavelets are requested but not implemented
        if self.feature_strategy == "wavelet":
            # Return deterministic zeros to keep shapes consistent until implemented
            return np.zeros((X.shape[0], X.shape[1] * len(self.BAND_RANGES)))
        # Raise an error for unsupported strategies to aid debugging
        raise ValueError(f"Unsupported feature strategy: {self.feature_strategy}")

    def _compute_fft_features(self, X):
        # Compute frequency bins corresponding to the real FFT for band selection
        freqs = np.fft.rfftfreq(X.shape[2], d=1.0 / self.sfreq)
        # Compute power spectra for each sample and channel
        power = np.abs(np.fft.rfft(X, axis=2)) ** 2
        # Prepare a container for aggregated band power per channel
        features = []
        # Iterate over predefined bands to accumulate mean power values
        for band in self.band_labels:
            # Extract band limits to identify relevant FFT bins
            low, high = self.BAND_RANGES[band]
            # Build a mask selecting frequencies inside the target band
            band_mask = (freqs >= low) & (freqs <= high)
            # Average power across the selected bins to summarize the band energy
            band_power = power[:, :, band_mask].mean(axis=2)
            # Append flattened band power per channel to the global feature set
            features.append(band_power)
        # Concatenate all band features along the channel axis to form the design matrix
        return np.concatenate(features, axis=1)


def extract_features(X, sfreq: float, feature_strategy: str = "fft", normalize: bool = True):
    """Helper to compute features without instantiating the transformer manually."""

    # Instantiate the transformer with the provided configuration for convenience
    transformer = ExtractFeatures(sfreq=sfreq, feature_strategy=feature_strategy, normalize=normalize)
    # Return the transformed feature matrix for immediate consumption
    return transformer.transform(X)
