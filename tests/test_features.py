"""Tests for configurable EEG feature extraction."""

# Importe time pour mesurer le temps d'exécution
import time

# Importe pytest pour vérifier les erreurs levées par l'extracteur
import pytest

# Importe numpy pour générer des signaux synthétiques contrôlés
import numpy as np

# Importe mne pour construire des Epochs simulant des enregistrements EEG
from mne import EpochsArray, create_info

# Importe les API d'extraction pour vérifier la conformité des contrôles
from tpv.features import ExtractFeatures, extract_features

# Définit un budget temporel strict pour garantir des performances interactives
TIME_BUDGET_S = 0.05


def _build_epochs(
    n_epochs: int, n_channels: int, n_times: int, sfreq: float
) -> EpochsArray:
    """Create synthetic epochs with explicit channel names."""

    # Génère un bruit gaussien pour simuler des essais EEG variés
    data = np.random.default_rng(seed=42).standard_normal(
        (n_epochs, n_channels, n_times)
    )
    # Construit des noms de canaux explicites pour valider les étiquettes générées
    ch_names = [f"C{idx}" for idx in range(n_channels)]
    # Crée l'info MNE pour fournir la fréquence d'échantillonnage et les noms de canaux
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    # Bâtit les epochs MNE afin de respecter l'API attendue par l'extracteur
    return EpochsArray(data, info)


def test_extract_features_welch_shape_and_labels() -> None:
    """Welch extraction should return flattened bands aligned with labels."""

    # Prépare des epochs synthétiques pour contrôler la dimension de sortie
    epochs = _build_epochs(n_epochs=2, n_channels=3, n_times=256, sfreq=128.0)
    # Lance l'extraction avec la configuration par défaut basée sur Welch
    features, labels = extract_features(epochs, config={"method": "welch"})
    # Vérifie que la matrice contient une colonne par paire canal-bande
    assert features.shape == (2, 12)
    # Confirme que l'ordre des étiquettes reflète les canaux puis les bandes
    assert labels[:4] == ["C0_theta", "C0_alpha", "C0_beta", "C0_gamma"]
    # Garantit que chaque canal génère le même nombre d'étiquettes de bande
    assert len(labels) == features.shape[1]


def test_extract_features_wavelet_placeholder() -> None:
    """Wavelet mode must provide a zero placeholder while keeping labels."""

    # Prépare des epochs synthétiques pour tester la voie placeholder
    epochs = _build_epochs(n_epochs=1, n_channels=2, n_times=128, sfreq=256.0)
    # Demande explicitement la méthode wavelet pour obtenir un tenseur nul
    features, labels = extract_features(epochs, config={"method": "wavelet"})
    # Vérifie que les features sont bien nuls faute d'implémentation
    assert np.array_equal(features, np.zeros((1, 8)))
    # Contrôle que les étiquettes restent alignées avec la matrice retournée
    assert labels == [
        "C0_theta",
        "C0_alpha",
        "C0_beta",
        "C0_gamma",
        "C1_theta",
        "C1_alpha",
        "C1_beta",
        "C1_gamma",
    ]


def test_extract_features_runtime_budget() -> None:
    """Welch extraction should respect the latency budget on small batches."""

    # Prépare un petit lot d'epochs pour mesurer la performance
    epochs = _build_epochs(n_epochs=5, n_channels=4, n_times=512, sfreq=128.0)
    # Capture l'instant de début pour mesurer la durée de l'extraction
    start = time.perf_counter()
    # Exécute l'extraction Welch avec les paramètres par défaut
    extract_features(epochs, config={"method": "welch"})
    # Capture l'instant de fin pour comparer avec le budget
    stop = time.perf_counter()
    # Garantit que le traitement reste inférieur au budget temporel fixé
    assert (stop - start) < TIME_BUDGET_S


def test_extract_features_transform_rejects_wrong_shape() -> None:
    """The scikit-learn wrapper should guard against malformed tensors."""

    # Construit un tenseur 2D pour déclencher la validation des dimensions
    bad_tensor = np.zeros((4, 16))
    # Instancie l'extracteur avec une fréquence arbitraire pour le test
    extractor = ExtractFeatures(sfreq=128.0)
    # Vérifie que la validation signale l'absence de la troisième dimension
    with pytest.raises(ValueError):
        extractor.transform(bad_tensor)


def test_extract_features_transform_rejects_unknown_strategy() -> None:
    """The wrapper must refuse unsupported feature strategies."""

    # Crée un tenseur conforme pour isoler l'erreur de stratégie
    eeg_tensor = np.zeros((1, 1, 16))
    # Force une stratégie inconnue afin d'atteindre la branche d'erreur
    extractor = ExtractFeatures(sfreq=128.0, feature_strategy="unknown")
    # Confirme que l'appelant est informé de la stratégie non supportée
    with pytest.raises(ValueError):
        extractor.transform(eeg_tensor)


def test_extract_features_transform_wavelet_placeholder() -> None:
    """The wrapper should mirror the procedural placeholder for wavelets."""

    # Génère un tenseur factice pour tester la branche placeholder
    eeg_tensor = np.ones((2, 3, 32))
    # Configure l'extracteur pour utiliser la stratégie wavelet non implémentée
    extractor = ExtractFeatures(sfreq=128.0, feature_strategy="wavelet")
    # Vérifie que le placeholder respecte la forme attendue sans calcul réel
    output = extractor.transform(eeg_tensor)
    # Garantit que le placeholder renvoie des zéros dimensionnés par bande
    assert np.array_equal(output, np.zeros((2, 12)))


def test_extract_features_rejects_unknown_method() -> None:
    """The procedural helper should reject unhandled extraction methods."""

    # Prépare des epochs pour activer la voie de validation de méthode
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=64, sfreq=64.0)
    # Passe une méthode inconnue pour vérifier le message d'erreur explicite
    with pytest.raises(ValueError):
        extract_features(epochs, config={"method": "unknown"})
