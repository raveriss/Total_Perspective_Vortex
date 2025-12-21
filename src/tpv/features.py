"""Feature extraction utilities for EEG signals."""

# Importe les annotations pour clarifier la signature des fonctions
# Importe Any pour typer la configuration dynamique des fonctions
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, cast

# Importe NumPy pour manipuler les tenseurs spectraux et tabulaires
import numpy as np

# Importe scipy.signal pour accéder à l'estimateur de Welch et à la CWT
from scipy import signal

# Importe BaseEstimator et TransformerMixin pour conserver la compatibilité scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin

# Nombre attendu de bornes (basse, haute) pour chaque bande de fréquence
BAND_BOUND_COUNT = 2


def _resolve_band_ranges(
    config: Mapping[str, Any], default_bands: Mapping[str, Tuple[float, float]]
) -> Dict[str, Tuple[float, float]]:
    """Retourne les bandes explicites en préservant l'ordre demandé."""

    # Fusionne les bandes personnalisées pour respecter l'ordre d'entrée
    configured_bands = config.get("bands", None)
    if configured_bands is None:
        return _validate_band_mapping(default_bands)
    return _validate_band_mapping(configured_bands)


def _coerce_band_bounds(bounds: Sequence[float]) -> Tuple[float, float]:
    """Convertit et valide les bornes numériques d'une bande."""

    if not isinstance(bounds, Sequence) or len(bounds) != BAND_BOUND_COUNT:
        raise ValueError("Band ranges must be (low, high) pairs.")
    try:
        low, high = (float(bounds[0]), float(bounds[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError("Band edges must be numeric.") from exc
    return low, high


def _validate_band_edges(low: float, high: float) -> None:
    """Valide la cohérence des bornes de fréquence."""

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Band edges must be finite.")
    if low < 0 or high <= low:
        raise ValueError("Band edges must satisfy 0 <= low < high.")


def _validate_single_band(
    name: str, bounds: Sequence[float]
) -> Tuple[str, Tuple[float, float]]:
    """Valide un couple (nom, bornes) et retourne une version normalisée."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Band names must be non-empty strings.")
    low, high = _coerce_band_bounds(bounds)
    _validate_band_edges(low, high)
    return name.strip(), (low, high)


def _validate_band_mapping(
    bands: (
        Mapping[str, Tuple[float, float]] | Iterable[Tuple[str, Tuple[float, float]]]
    ),
) -> Dict[str, Tuple[float, float]]:
    """Valide les bandes fréquence et garantit une copie défensive."""

    resolved = dict(bands)
    if not resolved:
        raise ValueError("At least one frequency band must be provided.")
    validated: Dict[str, Tuple[float, float]] = {}
    for name, bounds in resolved.items():
        validated_name, validated_bounds = _validate_single_band(name, bounds)
        validated[validated_name] = validated_bounds
    return validated


def _prepare_welch_parameters(
    config: Mapping[str, Any], n_times: int
) -> Tuple[str | Iterable[float], int, int | None, str, str]:
    """Calcule des paramètres Welch bornés pour éviter les avertissements."""

    # Applique une fenêtre lisse pour limiter les fuites fréquentielles
    window: str | Iterable[float] = _validate_welch_window(config.get("window", "hann"))
    # Permet d'ajuster la taille de segment pour contrôler la résolution
    effective_nperseg = _sanitize_nperseg(config.get("nperseg"), n_times)
    # Offre un recouvrement configurable pour stabiliser l'estimation
    effective_noverlap = _sanitize_noverlap(
        config.get("noverlap"), effective_nperseg=effective_nperseg
    )
    # Permet de choisir la stratégie d'agrégation des segments
    average: str = config.get("average", "mean")
    # Permet de choisir la densité ou la puissance intégrée
    scaling: str = config.get("scaling", "density")
    # Regroupe les paramètres bornés pour l'appel Welch
    return window, effective_nperseg, effective_noverlap, average, scaling


def _validate_wavelet_width(wavelet_width: Any) -> float:
    """Valide la largeur de wavelet pour éviter une énergie dégénérée."""

    try:
        width = float(wavelet_width)
    except (TypeError, ValueError) as exc:
        raise ValueError("wavelet_width must be a positive float.") from exc
    if not np.isfinite(width) or width <= 0:
        raise ValueError("wavelet_width must be a positive float.")
    return width


def _validate_welch_window(window: str | Iterable[float]) -> str | Iterable[float]:
    """Valide la fenêtre Welch pour éviter des appels non déterministes."""

    if isinstance(window, str):
        if not window.strip():
            raise ValueError("Welch window name must be a non-empty string.")
        return window
    if isinstance(window, Iterable):
        try:
            window_tuple: Tuple[float, ...] = tuple(float(value) for value in window)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Welch window iterable must contain numeric values."
            ) from exc
        if len(window_tuple) == 0:
            raise ValueError("Welch window iterable cannot be empty.")
        return window_tuple
    raise ValueError("Welch window must be a string or an iterable.")


def _sanitize_nperseg(nperseg: Any, n_times: int) -> int:
    """Valide et borne nperseg pour conserver une fenêtre positive."""

    if nperseg is None:
        return n_times
    if not isinstance(nperseg, int):
        raise ValueError("Welch nperseg must be an integer or None.")
    if nperseg <= 0:
        return n_times
    return min(nperseg, n_times)


def _sanitize_noverlap(noverlap: Any, *, effective_nperseg: int) -> int | None:
    """Valide le recouvrement en conservant une fenêtre strictement positive."""

    if noverlap is None:
        return None
    if not isinstance(noverlap, int):
        raise ValueError("Welch noverlap must be a non-negative integer or None.")
    return max(0, min(noverlap, effective_nperseg - 1))


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


def _compute_wavelet_coefficients(
    channel_values: np.ndarray,
    central_frequencies: Sequence[float],
    sfreq: float,
    wavelet_cycles: float,
) -> np.ndarray:
    """Calcule les coefficients wavelets en convoluant une gaussienne modulée."""

    # Prépare un conteneur pour stocker les coefficients par échelle
    coefficients: List[np.ndarray] = []
    # Prépare un axe temporel centré pour construire la gaussienne
    centered_times = np.arange(channel_values.size) - (channel_values.size - 1) / 2
    # Parcourt les fréquences centrales pour projeter le signal
    for central_frequency in central_frequencies:
        # Empêche une fréquence nulle pour éviter des divisions par zéro
        safe_frequency = max(central_frequency, 1e-9)
        # Calcule l'écart-type de la gaussienne en fonction du nombre de cycles
        sigma = wavelet_cycles * sfreq / safe_frequency
        # Construit la gaussienne centrée pour limiter les fuites temporelles
        envelope = np.exp(-(centered_times**2) / (2 * sigma**2))
        # Construit l'oscillation complexe alignée sur la fréquence centrale
        oscillation = np.exp(2j * np.pi * safe_frequency * centered_times / sfreq)
        # Combine l'enveloppe et l'oscillation pour former la wavelet
        wavelet = envelope * oscillation
        # Convolue le signal avec la wavelet via FFT pour l'efficacité
        convolved = signal.fftconvolve(channel_values, wavelet, mode="same")
        # Ajoute le résultat à la liste pour assembler la matrice finale
        coefficients.append(convolved)
    # Empile les coefficients en matrice (bandes, temps) pour analyse d'énergie
    return np.stack(coefficients, axis=0)


def _compute_wavelet_band_powers(
    data: np.ndarray,
    sfreq: float,
    band_ranges: Mapping[str, Tuple[float, float]],
    config: Mapping[str, Any],
) -> np.ndarray:
    """Calcule l'énergie de bandes via une CWT Morlet paramétrable."""

    # Valide le nom de wavelet demandé pour éviter les comportements implicites
    wavelet_name = config.get("wavelet", "morlet")
    if wavelet_name != "morlet":
        raise ValueError(f"Unsupported wavelet: {wavelet_name!r}. Use 'morlet'.")
    # Configure la largeur de la wavelet pour ajuster la résolution temps-fréquence
    wavelet_cycles: float = _validate_wavelet_width(config.get("wavelet_width", 6.0))
    # Calcule la fréquence centrale de chaque bande pour cibler la wavelet
    central_frequencies: List[float] = [
        (low + high) / 2.0 for low, high in band_ranges.values()
    ]
    # Prépare un tableau pour stocker l'énergie par essai, canal et bande
    band_powers: np.ndarray = np.zeros((data.shape[0], data.shape[1], len(band_ranges)))
    # Parcourt chaque essai pour éviter des allocations massives inutiles
    for epoch_index, epoch_data in enumerate(data):
        # Parcourt chaque canal pour projeter le signal sur les échelles ciblées
        for channel_index, channel_values in enumerate(epoch_data):
            # Calcule les coefficients CWT sur les échelles demandées
            coefficients = _compute_wavelet_coefficients(
                channel_values,
                central_frequencies,
                sfreq,
                wavelet_cycles,
            )
            # Calcule la puissance moyenne par bande en intégrant la magnitude
            band_energy = np.abs(coefficients) ** 2
            # Moyenne temporelle pour stabiliser l'énergie de chaque bande
            band_powers[epoch_index, channel_index, :] = band_energy.mean(axis=1)
    # Retourne le tenseur énergie pour alignement avec les étiquettes de bandes
    return band_powers


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
        self,
        sfreq: float,
        feature_strategy: str = "fft",
        normalize: bool = True,
        bands: Mapping[str, Tuple[float, float]] | None = None,
        strategy_config: Mapping[str, Any] | None = None,
    ):
        # Stocke la fréquence d'échantillonnage comme float pour la FFT
        self.sfreq = float(sfreq)
        # Sélectionne la stratégie d'extraction ("fft" ou "wavelet")
        self.feature_strategy = feature_strategy
        # Active ou non la normalisation des features
        self.normalize = normalize
        # Expose les bandes pour compatibilité scikit-learn (get_params)
        self.bands = bands
        # Stocke les bandes utilisées pour la construction des features
        self.band_ranges: Dict[str, Tuple[float, float]] = _validate_band_mapping(
            bands if bands is not None else self.BAND_RANGES
        )
        # Stocke la configuration spécifique à la stratégie (Welch, wavelet, etc.)
        self.strategy_config = strategy_config
        strategy_payload: Dict[str, Any] = dict(strategy_config or {})
        self._effective_strategy_config: Dict[str, Any] = dict(strategy_payload)

    def fit(self, X, y=None):
        # Pas d'apprentissage de paramètres pour l'instant
        return self

    def transform(self, X):
        # Vérifie que X est bien (n_epochs, n_channels, n_times)
        if X.ndim != self.EXPECTED_EEG_NDIM:
            raise ValueError("X must have shape (n_samples, n_channels, n_times)")

        # Calcule les features brutes selon la stratégie choisie
        raw_features = self._compute_features(X)

        # Normalise optionnellement les features bande-par-bande
        if self.normalize:
            mean = raw_features.mean(axis=1, keepdims=True)
            std = raw_features.std(axis=1, keepdims=True) + self.NORMALIZATION_EPS
            features = (raw_features - mean) / std
        else:
            features = raw_features

        # Vérifie qu'aucune valeur non finie ne subsiste
        if not np.all(np.isfinite(features)):
            raise ValueError(
                "ExtractFeatures produced non-finite values (NaN/Inf). "
                "Check band ranges and sampling frequency."
            )

        return features

    @property
    def band_labels(self) -> List[str]:
        # Retourne les noms de bandes dans l'ordre déclaré
        return list(self.band_ranges.keys())

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        """Dispatch interne vers la bonne stratégie de features."""

        # Utilise la FFT comme stratégie par défaut
        if self.feature_strategy == "fft":
            return self._compute_fft_features(X)
        # Permet de basculer vers la stratégie wavelet
        if self.feature_strategy == "wavelet":
            return self._compute_wavelet_features(X)
        # Calcule les puissances de bandes via Welch
        if self.feature_strategy == "welch":
            return self._compute_welch_features(X)
        # Rejette explicitement les stratégies inconnues
        raise ValueError(
            f"Unsupported feature_strategy: {self.feature_strategy!r}. "
            "Use 'fft', 'welch', or 'wavelet'."
        )

    def _compute_fft_features(self, X: np.ndarray) -> np.ndarray:
        """Calcule les puissances de bandes à partir de la FFT."""

        # Calcule les fréquences réelles à partir de la sfreq configurée
        freqs = np.fft.rfftfreq(X.shape[2], d=1.0 / self.sfreq)
        # Calcule la puissance spectrale par canal et échantillon
        power = np.abs(np.fft.rfft(X, axis=2)) ** 2  # pragma: no mutate
        # Accumule les puissances par bande pour chaque canal
        band_powers: List[np.ndarray] = []
        # Parcourt chaque bande EEG définie dans les paramètres
        for _, (low, high) in self.band_ranges.items():
            # Construit le masque fréquentiel pour cette bande
            band_mask = (freqs >= low) & (freqs <= high)

            # Gère le cas où aucune fréquence ne tombe dans la bande
            if not np.any(band_mask):
                # Retourne un bloc de zéros pour préserver la forme
                band_power = np.zeros(power.shape[:2])
            else:
                # Moyenne la puissance sur les fréquences de la bande
                band_power = power[:, :, band_mask].mean(axis=2)

            # Ajoute la matrice (n_samples, n_channels) à la liste
            band_powers.append(band_power)

        # Empile les bandes pour retourner un tenseur cohérent canal x bande
        stacked = np.stack(band_powers, axis=2)
        # Retourne une matrice aplatie (échantillons, canaux * bandes)
        return cast(np.ndarray, stacked.reshape(stacked.shape[0], -1))

    def _compute_wavelet_features(self, X: np.ndarray) -> np.ndarray:
        """Calcule des features à partir de la CWT wavelet."""

        # Calcule les coefficients puis la puissance de bande via la fonction dédiée
        stacked = _compute_wavelet_band_powers(
            X,
            self.sfreq,
            self.band_ranges,
            self._effective_strategy_config,
        )

        # Retourne la matrice tabulaire prête pour un classifieur scikit-learn
        return stacked.reshape(stacked.shape[0], -1)

    def _compute_welch_features(self, X: np.ndarray) -> np.ndarray:
        """Calcule des features à partir de la méthode de Welch."""

        stacked = _compute_welch_band_powers(
            X,
            self.sfreq,
            self.band_ranges,
            self._effective_strategy_config,
        )
        return stacked.reshape(stacked.shape[0], -1)


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
    # Refuse une configuration vide pour éviter un empilement sans bandes
    if not band_ranges:
        raise ValueError("At least one frequency band must be provided.")
    # Récupère la méthode pour aiguiller la stratégie d'extraction
    method: str = effective_config.pop("method", "welch")
    # Permet de contrôler l'activation de la normalisation via la configuration
    normalize = bool(effective_config.pop("normalize", False))
    # Supprime la configuration des bandes pour ne pas dupliquer l'information
    effective_config.pop("bands", None)
    # Capture les noms de canaux pour aligner les étiquettes de features
    channel_names: Sequence[str] = getattr(epochs.info, "ch_names", [])
    # Extrait les données temporelles pour calculer les caractéristiques spectrales
    data: np.ndarray = epochs.get_data()
    # Récupère la fréquence d'échantillonnage indispensable au calcul fréquentiel
    sfreq: float = float(epochs.info["sfreq"])
    # Construit un extracteur scikit-learn pour déléguer le calcul
    estimator = ExtractFeatures(
        sfreq=sfreq,
        feature_strategy=method,
        normalize=normalize,
        bands=band_ranges,
        strategy_config=effective_config,
    )
    # Calcule les features en déléguant à l'extracteur scikit-learn
    features = estimator.transform(data)
    # Détermine le nombre de bandes pour reconstruire les étiquettes
    n_bands = len(band_ranges)
    # Recompose un tenseur factice pour générer les étiquettes canal/bande
    stacked = features.reshape(features.shape[0], data.shape[1], n_bands)
    # Construit des étiquettes canal_bande pour interpréter les features
    labels: List[str] = _build_labels(stacked, band_ranges, channel_names)
    # Retourne les features tabulaires accompagnés de leurs étiquettes
    return features, labels
