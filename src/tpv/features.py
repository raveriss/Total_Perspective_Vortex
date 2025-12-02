"""Feature extraction utilities for EEG signals."""

# Importe NumPy pour manipuler les tenseurs spectraux et tabulaires
import numpy as np
# Importe scipy.signal pour accéder à l'estimateur de Welch
from scipy import signal
# Importe les annotations pour clarifier la signature des fonctions
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
# Importe BaseEstimator et TransformerMixin pour conserver la compatibilité scikit-learn
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
        power = np.abs(np.fft.rfft(X, axis=2)) ** 2  # pragma: no mutate
        features = []
        for band in self.band_labels:
            low, high = self.BAND_RANGES[band]
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = power[:, :, band_mask].mean(axis=2)
            features.append(band_power)
        return np.concatenate(features, axis=1)


def extract_features(
    epochs: Any,
    config: Mapping[str, Any] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute band power features from epochs with configurable PSD options."""

    # Stocke une configuration vide lorsque l'appelant ne fournit rien
    effective_config: Dict[str, Any] = dict(config or {})
    # Déclare les bandes EEG usuelles pour guider l'extraction
    default_bands: Dict[str, Tuple[float, float]] = {
        "theta": (4.0, 7.0),
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
        "gamma": (31.0, 45.0),
    }
    # Fusionne les bandes personnalisées en conservant l'ordre demandé
    band_ranges: Dict[str, Tuple[float, float]] = effective_config.get(
        "bands", default_bands
    )
    # Récupère la méthode pour aiguiller la stratégie d'extraction
    method: str = effective_config.get("method", "welch")
    # Capture les noms de canaux pour aligner les étiquettes de features
    channel_names: Sequence[str] = getattr(epochs.info, "ch_names", [])
    # Extrait les données temporelles pour calculer les caractéristiques spectrales
    data: np.ndarray = epochs.get_data()
    # Récupère la fréquence d'échantillonnage indispensable au calcul fréquentiel
    sfreq: float = float(epochs.info["sfreq"])
    # Oriente vers le calcul Welch lorsque la méthode par défaut est demandée
    if method == "welch":
        # Applique une fenêtre lisse pour limiter les fuites fréquentielles
        window: str | Iterable[float] = effective_config.get("window", "hann")
        # Permet d'ajuster la taille de segment pour contrôler la résolution
        nperseg: int | None = effective_config.get("nperseg")
        # Offre un recouvrement configurable pour stabiliser l'estimation
        noverlap: int | None = effective_config.get("noverlap")
        # Calcule la densité spectrale de puissance par canal et par essai
        freqs, psd = signal.welch(
            data,
            sfreq,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=-1,
            average="mean",
        )
        # Accumule les puissances de bande pour chaque intervalle demandé
        band_powers: List[np.ndarray] = []
        # Parcourt les bandes dans l'ordre pour garantir la stabilité des colonnes
        for _, (low, high) in band_ranges.items():
            # Construit un masque fréquentiel pour isoler l'intervalle cible
            band_mask = (freqs >= low) & (freqs <= high)
            # Moyenne la PSD sur la bande pour réduire la dimension temporelle
            band_powers.append(psd[:, :, band_mask].mean(axis=-1))
        # Empile les bandes pour conserver la structure epochs x canaux x bandes
        stacked = np.stack(band_powers, axis=2)
    # Fournit un placeholder neutre pour les méthodes non encore implémentées
    elif method == "wavelet":
        # Calcule le nombre de bandes afin d'aligner le placeholder
        n_bands = len(band_ranges)
        # Génère un tenseur nul pour respecter l'API sans calcul réel
        stacked = np.zeros((data.shape[0], data.shape[1], n_bands))
    # Rejette explicitement les méthodes non reconnues pour éviter des surprises
    else:
        # Signale l'erreur avec la méthode fournie par l'appelant
        raise ValueError(f"Unsupported feature extraction method: {method}")
    # Prépare les étiquettes par canal et bande pour interpréter les colonnes
    labels: List[str] = []
    # Parcourt les canaux pour associer les bandes à chaque série temporelle
    for channel_index in range(stacked.shape[1]):
        # Sélectionne un nom explicite ou construit un identifiant générique
        channel_label = (
            channel_names[channel_index]
            if channel_names
            else f"ch{channel_index}"
        )
        # Ajoute une étiquette pour chaque bande afin de suivre l'ordre des colonnes
        for band_name in band_ranges.keys():
            # Concatène le canal et la bande pour un suivi lisible
            labels.append(f"{channel_label}_{band_name}")
    # Aplati les bandes pour fournir une matrice compatible avec scikit-learn
    flattened = stacked.reshape(stacked.shape[0], -1)
    # Retourne les features tabulaires accompagnés de leurs étiquettes
    return flattened, labels
