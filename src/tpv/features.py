"""Feature extraction utilities for EEG signals."""

# Importe les annotations pour clarifier la signature des fonctions
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# Importe NumPy pour manipuler les tenseurs spectraux et tabulaires
import numpy as np

# Importe scipy.signal pour accéder à l'estimateur de Welch
from scipy import signal

# Importe BaseEstimator et TransformerMixin pour conserver la compatibilité scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin


def _resolve_band_ranges(
    config: Mapping[str, Any], default_bands: Mapping[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """Retourne les bandes explicites en préservant l'ordre demandé."""

    # Fusionne les bandes personnalisées pour respecter l'ordre d'entrée
    return dict(config.get("bands", default_bands))


def _prepare_welch_parameters(
    config: Mapping[str, Any], n_times: int
) -> Tuple[str | Iterable[float], int, int | None, str, str]:
    """Calcule des paramètres Welch bornés pour éviter les avertissements."""

    # Applique une fenêtre lisse pour limiter les fuites fréquentielles
    window: str | Iterable[float] = config.get("window", "hann")
    # Permet d'ajuster la taille de segment pour contrôler la résolution
    nperseg: int | None = config.get("nperseg")
    # Borne la taille de segment pour éviter les avertissements SciPy
    effective_nperseg: int = min(nperseg or n_times, n_times)
    # Offre un recouvrement configurable pour stabiliser l'estimation
    noverlap: int | None = config.get("noverlap")
    # Borne le recouvrement pour garantir une fenêtre strictement positive
    effective_noverlap: int | None = None
    # Vérifie que l'appelant a fourni un recouvrement explicite
    if noverlap is not None:
        # Coupe le recouvrement juste avant la taille de fenêtre autorisée
        effective_noverlap = min(noverlap, effective_nperseg - 1)
    # Permet de choisir la stratégie d'agrégation des segments
    average: str = config.get("average", "mean")
    # Permet de choisir la densité ou la puissance intégrée
    scaling: str = config.get("scaling", "density")
    # Regroupe les paramètres bornés pour l'appel Welch
    return window, effective_nperseg, effective_noverlap, average, scaling


def _compute_welch_band_powers(
    data: np.ndarray,
    sfreq: float,
    band_ranges: Mapping[str, Tuple[float, float]],
    config: Mapping[str, Any],
) -> np.ndarray:
    """Calcule les puissances de bandes via Welch avec bornes sécurisées."""

    # Stocke le nombre d'échantillons pour calibrer les fenêtres de Welch
    n_times: int = data.shape[-1]
    # Calcule les paramètres Welch pour un appel robuste
    window, effective_nperseg, effective_noverlap, average, scaling = (
        _prepare_welch_parameters(config, n_times)
    )
    # Calcule la densité spectrale de puissance par canal et par essai
    freqs, psd = signal.welch(
        data,
        sfreq,
        window=window,
        nperseg=effective_nperseg,
        noverlap=effective_noverlap,
        axis=-1,
        average=average,
        scaling=scaling,
    )
    # Accumule les puissances de bande pour chaque intervalle demandé
    band_powers: List[np.ndarray] = []
    # Parcourt les bandes dans l'ordre pour garantir la stabilité des colonnes
    for _, (low, high) in band_ranges.items():
        # Construit un masque fréquentiel pour isoler l'intervalle cible
        band_mask = (freqs >= low) & (freqs <= high)
        # Renvoie des zéros si aucune fréquence n'est disponible dans la bande
        if not np.any(band_mask):
            # Fournit un tenseur nul pour préserver la forme de sortie
            band_powers.append(np.zeros(psd.shape[:2]))
        else:
            # Moyenne la PSD sur la bande pour réduire la dimension temporelle
            band_powers.append(psd[:, :, band_mask].mean(axis=-1))
    # Empile les bandes pour conserver la structure epochs x canaux x bandes
    return np.stack(band_powers, axis=2)


def _placeholder_wavelet(data: np.ndarray, n_bands: int) -> np.ndarray:
    """Retourne un tenseur nul pour le mode wavelet non implémenté."""

    # Génère un tenseur nul pour respecter l'API sans calcul réel
    return np.zeros((data.shape[0], data.shape[1], n_bands))


def _build_labels(
    stacked: np.ndarray,
    band_ranges: Mapping[str, Tuple[float, float]],
    channel_names: Sequence[str],
) -> List[str]:
    """Construit des étiquettes canal_bande pour interpréter les features."""

    # Prépare les étiquettes par canal et bande pour interpréter les colonnes
    labels: List[str] = []
    # Parcourt les canaux pour associer les bandes à chaque série temporelle
    for channel_index in range(stacked.shape[1]):
        # Sélectionne un nom explicite ou construit un identifiant générique
        channel_label = (
            channel_names[channel_index] if channel_names else f"ch{channel_index}"
        )
        # Ajoute une étiquette pour chaque bande afin de suivre l'ordre des colonnes
        for band_name in band_ranges.keys():
            # Concatène le canal et la bande pour un suivi lisible
            labels.append(f"{channel_label}_{band_name}")
    # Retourne la liste d'étiquettes alignée sur la matrice aplatie
    return labels


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
    band_ranges: Dict[str, Tuple[float, float]] = _resolve_band_ranges(
        effective_config, default_bands
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
        # Calcule les PSD de bandes via Welch avec paramètres bornés
        stacked = _compute_welch_band_powers(data, sfreq, band_ranges, effective_config)
    # Fournit un placeholder neutre pour les méthodes non encore implémentées
    elif method == "wavelet":
        # Génère un tenseur nul pour respecter l'API sans calcul réel
        stacked = _placeholder_wavelet(data, len(band_ranges))
    # Rejette explicitement les méthodes non reconnues pour éviter des surprises
    else:
        # Signale l'erreur avec la méthode fournie par l'appelant
        raise ValueError(f"Unsupported feature extraction method: {method}")
    # Construit des étiquettes canal_bande pour interpréter les features
    labels: List[str] = _build_labels(stacked, band_ranges, channel_names)
    # Aplati les bandes pour fournir une matrice compatible avec scikit-learn
    flattened = stacked.reshape(stacked.shape[0], -1)
    # Retourne les features tabulaires accompagnés de leurs étiquettes
    return flattened, labels
