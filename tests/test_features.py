# Importe numpy pour produire des signaux synthétiques contrôlés
import numpy as np

# Importe pytest pour les assertions numériques tolérantes
import pytest

# Importe time pour suivre le budget temporel d'extraction
from time import perf_counter

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
    # Instancie la classe pour couvrir le chemin wavelet interne
    extractor = ExtractFeatures(sfreq=64.0, feature_strategy="wavelet")
    # Applique transform pour déclencher le placeholder de la classe
    transformed = extractor.transform(epochs.get_data())
    # Vérifie que le placeholder renvoie aussi des zéros
    assert np.array_equal(transformed, np.zeros((1, 12)))


def test_extract_features_wrapper_rejects_unknown_strategy() -> None:
    """La classe scikit-learn doit refuser les stratégies non supportées."""

    # Construit un tenseur conforme pour l'appel
    tensor = np.zeros((1, 2, 16))
    # Instancie un extracteur avec une stratégie inconnue
    extractor = ExtractFeatures(sfreq=128.0, feature_strategy="invalid")
    # Vérifie que l'appelant reçoit une erreur explicite
    with pytest.raises(ValueError):
        extractor.transform(tensor)


def test_extract_features_wrapper_rejects_incorrect_shape() -> None:
    """La classe scikit-learn doit refuser une dimension d'entrée erronée."""

    # Prépare un extracteur pour tester la validation de dimension
    extractor = ExtractFeatures(sfreq=128.0)
    # Prépare un tableau bidimensionnel pour déclencher la validation
    bad_shape = np.zeros((2, 16))
    # Vérifie que la méthode transforme lève une erreur de forme
    with pytest.raises(ValueError):
        extractor.transform(bad_shape)


def test_extract_features_wrapper_rejects_unknown_method() -> None:
    """La fonction procédurale doit refuser une méthode non supportée."""

    # Prépare des epochs minimaux pour l'appel procédural
    epochs = _build_epochs(n_epochs=1, n_channels=1, n_times=32, sfreq=64.0)
    # Vérifie que la fonction signale la méthode inconnue
    with pytest.raises(ValueError):
        extract_features(epochs, config={"method": "invalid"})


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


def test_extract_features_respects_time_budget() -> None:
    """L'extraction Welch doit rester sous un budget temps raisonnable."""

    # Construit des epochs plus longs pour sonder la performance temporelle
    epochs = _build_epochs(n_epochs=5, n_channels=4, n_times=512, sfreq=256.0)
    # Mesure l'instant initial pour contrôler la durée d'extraction
    start = perf_counter()
    # Lance l'extraction avec une configuration Welch recouvrante
    features, labels = extract_features(
        epochs, config={"method": "welch", "nperseg": 256, "noverlap": 128}
    )
    # Calcule la durée écoulée pour valider le budget temporel
    elapsed = perf_counter() - start
    # Vérifie que le nombre de colonnes correspond bien aux étiquettes
    assert features.shape[1] == len(labels)
    # Vérifie que l'extraction reste sous un quart de seconde
    assert elapsed < 0.25
