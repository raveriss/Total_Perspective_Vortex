# Importe numpy pour construire des signaux contrôlés et vérifier les résultats
import numpy as np

# Importe pytest pour capturer les erreurs et les assertions
import pytest

# Importe mne pour créer des epochs synthétiques conformes à l'API
from mne import EpochsArray, create_info

# Importe l'extracteur procédural et la classe scikit-learn associée
from tpv.features import ExtractFeatures, extract_features


def _build_epochs(
    n_epochs: int, n_channels: int, n_times: int, sfreq: float
) -> EpochsArray:
    """Crée des epochs synthétiques avec des noms de canal explicites."""

    # Génère un bruit gaussien pour éviter des signaux dégénérés
    data = np.random.default_rng(seed=0).standard_normal(
        (n_epochs, n_channels, n_times)
    )
    # Déclare des noms de canaux pour suivre l'ordre des étiquettes
    ch_names = [f"C{idx}" for idx in range(n_channels)]
    # Construit l'info MNE avec la fréquence d'échantillonnage imposée
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    # Retourne les epochs prêt à être traités par l'extracteur
    return EpochsArray(data, info)


def test_extract_features_welch_respects_window_and_shape() -> None:
    """Welch doit produire des bandes aplaties et étiquetées par canal."""

    # Prépare des epochs de test avec plusieurs canaux
    epochs = _build_epochs(n_epochs=3, n_channels=2, n_times=256, sfreq=128.0)
    # Demande une fenêtre rectangulaire pour stabiliser le test
    features, labels = extract_features(
        epochs,
        config={"method": "welch", "window": "boxcar", "nperseg": 128},
    )
    # Vérifie que la matrice est bien de taille essais x (canaux * bandes)
    assert features.shape == (3, 8)
    # Vérifie que les étiquettes reflètent l'ordre canal puis bande
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


def test_extract_features_alpha_sine_dominates_alpha_band() -> None:
    """Un sinus alpha doit produire plus d'énergie dans la bande alpha."""

    # Fixe les paramètres temporels pour centrer le sinus sur 10 Hz
    sfreq = 128.0
    # Construit un axe temporel régulier pour le sinus
    times = np.arange(0, 2.0, 1.0 / sfreq)
    # Génère un sinus de 10 Hz pour la première composante
    alpha_signal = np.sin(2 * np.pi * 10.0 * times)
    # Empile le sinus et du bruit sur deux canaux
    data = np.stack(
        [alpha_signal, np.random.default_rng(seed=1).standard_normal(times.size)],
    )
    # Réplique le signal sur plusieurs essais pour renforcer la moyenne
    epochs_data = np.stack([data, data])
    # Crée l'info correspondante pour MNE
    info = create_info(ch_names=["C0", "C1"], sfreq=sfreq, ch_types="eeg")
    # Bâtit les epochs à partir du tenseur préparé
    epochs = EpochsArray(epochs_data, info)
    # Exécute l'extraction avec la méthode par défaut
    features, _ = extract_features(epochs)
    # Reshape pour retrouver la structure essais x canaux x bandes
    reshaped = features.reshape(2, 2, 4)
    # Vérifie que le canal alpha présente une énergie maximale dans la bande alpha
    assert reshaped[0, 0, 1] == pytest.approx(np.max(reshaped[0, 0]))


def test_extract_features_wavelet_placeholder_preserves_shape() -> None:
    """La voie wavelet doit renvoyer des zéros avec la bonne dimension."""

    # Prépare des epochs synthétiques minimaux
    epochs = _build_epochs(n_epochs=1, n_channels=3, n_times=64, sfreq=64.0)
    # Demande explicitement la méthode wavelet non implémentée
    features, labels = extract_features(epochs, config={"method": "wavelet"})
    # Vérifie que la matrice est entièrement nulle
    assert np.array_equal(features, np.zeros((1, 12)))
    # Vérifie que les étiquettes restent cohérentes avec les canaux
    assert labels[0] == "C0_theta"


def test_extract_features_wrapper_rejects_unknown_strategy() -> None:
    """La classe scikit-learn doit refuser les stratégies non supportées."""

    # Construit un tenseur conforme pour l'appel
    tensor = np.zeros((1, 2, 16))
    # Instancie un extracteur avec une stratégie inconnue
    extractor = ExtractFeatures(sfreq=128.0, feature_strategy="invalid")
    # Vérifie que l'appelant reçoit une erreur explicite
    with pytest.raises(ValueError):
        extractor.transform(tensor)


def test_extract_features_numeric_stability() -> None:
    """Welch doit produire des valeurs finies et non négatives."""

    # Prépare un bruit blanc pour sonder la stabilité numérique
    epochs = _build_epochs(n_epochs=2, n_channels=1, n_times=512, sfreq=256.0)
    # Exécute l'extraction pour récupérer les PSD de bande
    features, _ = extract_features(epochs, config={"method": "welch"})
    # Vérifie l'absence de NaN ou d'infini
    assert np.isfinite(features).all()
    # Vérifie l'absence de valeurs négatives après la moyenne de puissance
    assert (features >= 0).all()
