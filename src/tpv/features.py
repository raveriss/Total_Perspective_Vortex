"""Feature extraction utilities for EEG signals."""

# Importe les annotations pour clarifier la signature des fonctions
# Importe Any pour typer la configuration dynamique des fonctions
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

# Importe NumPy pour manipuler les tenseurs spectraux et tabulaires
import numpy as np
from numpy.typing import NDArray

# Importe scipy.signal pour accéder à l'estimateur de Welch et à la CWT
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
    if nperseg is None or nperseg <= 0:
        effective_nperseg = n_times
    else:
        effective_nperseg = min(nperseg, n_times)
    # Offre un recouvrement configurable pour stabiliser l'estimation
    noverlap: int | None = config.get("noverlap")
    # Borne le recouvrement pour garantir une fenêtre strictement positive
    effective_noverlap: int | None = None
    # Vérifie que l'appelant a fourni un recouvrement explicite
    if noverlap is not None:
        # Coupe le recouvrement juste avant la taille de fenêtre autorisée
        effective_noverlap = max(0, min(noverlap, effective_nperseg - 1))
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


def _compute_wavelet_coefficients(  # noqa: PLR0913
    channel_values: np.ndarray,
    central_frequencies: Sequence[float],
    sfreq: float,
    wavelet_cycles: float,
    wavelet_name: str = "morlet",
    max_levels: int | None = None,
) -> np.ndarray:
    """Calcule les coefficients wavelets en convoluant une gaussienne modulée."""

    def _ricker_wavelet(points: int, width: float) -> NDArray[np.float64]:
        """Implémente une version locale de l'ondelette Ricker."""

        safe_width = max(width, 1e-9)
        positions = np.linspace(-(points - 1) / 2.0, (points - 1) / 2.0, points)
        normalized = positions / safe_width
        return (
            2.0
            / (np.sqrt(3.0 * safe_width) * np.pi**0.25)
            * (1.0 - normalized**2)
            * np.exp(-(normalized**2) / 2.0)
        )

    # Restreint le nombre de niveaux si une limite est fournie
    if max_levels is not None:
        if max_levels <= 0:
            raise ValueError("wavelet_max_level must be positive.")
        effective_frequencies = list(central_frequencies)[:max_levels]
    else:
        effective_frequencies = list(central_frequencies)
    # Vérifie qu'au moins une fréquence a été fournie
    if not effective_frequencies:
        raise ValueError("At least one wavelet frequency is required.")

    # Prépare un conteneur pour stocker les coefficients par échelle
    coefficients: List[np.ndarray] = []
    # Prépare un axe temporel centré pour construire la gaussienne
    centered_times = np.arange(channel_values.size) - (channel_values.size - 1) / 2
    # Parcourt les fréquences centrales pour projeter le signal
    for central_frequency in effective_frequencies:
        # Empêche une fréquence nulle pour éviter des divisions par zéro
        safe_frequency = max(central_frequency, 1e-9)
        # Calcule l'écart-type de la gaussienne en fonction du nombre de cycles
        sigma = wavelet_cycles * sfreq / safe_frequency
        # Sélectionne la forme d'onde mère à utiliser
        if wavelet_name == "morlet":
            # Construit la gaussienne centrée pour limiter les fuites temporelles
            envelope = np.exp(-(centered_times**2) / (2 * sigma**2))
            # Construit l'oscillation complexe alignée sur la fréquence centrale
            oscillation = np.exp(2j * np.pi * safe_frequency * centered_times / sfreq)
            # Combine l'enveloppe et l'oscillation pour former la wavelet
            wavelet = envelope * oscillation
        elif wavelet_name in {"ricker", "mexh"}:
            # Construit une ondelette de type chapeau mexicain (Ricker)
            wavelet = _ricker_wavelet(centered_times.size, sigma)
        else:
            raise ValueError(f"Unsupported wavelet name: {wavelet_name}")
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

    # Configure la largeur de la wavelet pour ajuster la résolution temps-fréquence
    wavelet_cycles: float = float(config.get("wavelet_width", 6.0))
    # Sélectionne la forme d'onde mère à utiliser
    wavelet_name: str = str(config.get("wavelet", "morlet"))
    # Calcule la fréquence centrale de chaque bande pour cibler la wavelet
    central_frequencies: List[float] = [
        (low + high) / 2.0 for low, high in band_ranges.values()
    ]
    # Borne le nombre de niveaux à calculer si demandé
    max_levels_config = config.get("wavelet_max_level")
    effective_levels = len(central_frequencies)
    if max_levels_config is not None:
        max_levels = int(max_levels_config)
        if max_levels <= 0:
            raise ValueError("wavelet_max_level must be positive.")
        effective_levels = min(max_levels, len(central_frequencies))
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
                wavelet_name=wavelet_name,
                max_levels=effective_levels,
            )
            # Calcule la puissance moyenne par bande en intégrant la magnitude
            band_energy = np.abs(coefficients) ** 2
            # Moyenne temporelle pour stabiliser l'énergie de chaque bande
            band_powers[epoch_index, channel_index, :effective_levels] = (
                band_energy.mean(axis=1)
            )
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
    ):
        # Stocke la fréquence d'échantillonnage comme float pour la FFT
        self.sfreq = float(sfreq)
        # Sélectionne la stratégie d'extraction ("fft" ou "wavelet")
        self.feature_strategy = feature_strategy
        # Active ou non la normalisation des features
        self.normalize = normalize

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
        return list(self.BAND_RANGES.keys())

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        """Dispatch interne vers la bonne stratégie de features."""

        # Utilise la FFT comme stratégie par défaut
        if self.feature_strategy == "fft":
            return self._compute_fft_features(X)
        # Permet de basculer vers la stratégie wavelet
        if self.feature_strategy == "wavelet":
            return self._compute_wavelet_features(X)
        # Rejette explicitement les stratégies inconnues
        raise ValueError(
            f"Unsupported feature_strategy: {self.feature_strategy!r}. "
            "Use 'fft' or 'wavelet'."
        )

    def _compute_fft_features(self, X: np.ndarray) -> np.ndarray:
        """Calcule les puissances de bandes à partir de la FFT."""

        # Calcule les fréquences réelles à partir de la sfreq configurée
        freqs = np.fft.rfftfreq(X.shape[2], d=1.0 / self.sfreq)
        # Calcule la puissance spectrale par canal et échantillon
        power = np.abs(np.fft.rfft(X, axis=2)) ** 2  # pragma: no mutate
        # Prépare le conteneur pour accumuler les puissances de bandes
        features: List[np.ndarray] = []

        # Parcourt chaque bande EEG définie dans BAND_RANGES
        for band in self.band_labels:
            # Récupère les bornes fréquentielles de la bande
            low, high = self.BAND_RANGES[band]
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
            features.append(band_power)

        # Concatène les bandes le long de l’axe des features
        return np.concatenate(features, axis=1)

    def _compute_wavelet_features(self, X: np.ndarray) -> np.ndarray:
        """Calcule des features à partir de la CWT wavelet."""

        # Calcule la fréquence centrale pour chaque bande prédéfinie
        central_frequencies = [
            (low + high) / 2.0 for low, high in self.BAND_RANGES.values()
        ]
        # Prépare une matrice vide pour accueillir les features wavelets
        features = np.zeros((X.shape[0], X.shape[1] * len(self.BAND_RANGES)))

        # Parcourt chaque essai pour limiter la charge mémoire
        for epoch_index, epoch_data in enumerate(X):
            # Parcourt chaque canal pour calculer les coefficients wavelets
            for channel_index, channel_values in enumerate(epoch_data):
                # Calcule la CWT pour toutes les bandes en une seule fois
                coefficients = _compute_wavelet_coefficients(
                    channel_values,
                    central_frequencies,
                    self.sfreq,
                    wavelet_cycles=6.0,
                    wavelet_name="morlet",
                    max_levels=len(central_frequencies),
                )
                # Calcule l'énergie moyenne par bande à partir des coefficients
                band_energy = np.abs(coefficients) ** 2
                # Aplatit le tenseur pour l'insérer dans la matrice finale
                start = channel_index * len(self.BAND_RANGES)
                end = start + len(self.BAND_RANGES)
                # Place l'énergie moyenne dans les colonnes associées au canal
                features[epoch_index, start:end] = band_energy.mean(axis=1)

        # Retourne la matrice tabulaire prête pour un classifieur scikit-learn
        return features


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
    # Oriente vers la transformée en ondelettes pour une résolution temporelle
    elif method == "wavelet":
        # Calcule l'énergie de bandes via une CWT configurée
        stacked = _compute_wavelet_band_powers(
            data, sfreq, band_ranges, effective_config
        )
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
